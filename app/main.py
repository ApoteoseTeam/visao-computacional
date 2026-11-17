import subprocess

import cv2
import json
import time
from pathlib import Path
from datetime import datetime
from collections import deque

import numpy as np

from detector import PersonDetector
from tracker import PersonTracker

ROOT = Path(__file__).resolve().parent.parent
VIDEO_PATH = ROOT / 'videos' / 'pessoas3.mp4'
MODEL_PATH = ROOT / 'models' / 'yolo11n_960.onnx'
CONFIG_PATH = ROOT / 'app' / 'gate_config.json'
OUTPUT_DIR = ROOT / 'outputs'
WINDOW = 'Reali | Calibracao do Portao'
CONFIDENCE = 0.25
DISPLAY_WIDTH = 1100
REF_W, REF_H = 1275, 717
DEFAULT_POINTS = {
    'inner_left': [535 / REF_W, 325 / REF_H],
    'inner_right': [745 / REF_W, 325 / REF_H],
    'outer_left': [505 / REF_W, 515 / REF_H],
    'outer_right': [755 / REF_W, 515 / REF_H],
}
HANDLE_RADIUS = 11
MAX_CROSSING_SECONDS = 10.0
TRACK_TIMEOUT_SECONDS = 3.0
GREEN, RED, YELLOW, BLUE, WHITE = (70, 220, 90), (70, 70, 240), (0, 220, 255), (255, 180, 40), (255, 255, 255)
KEYS = ('inner_left', 'inner_right', 'outer_left', 'outer_right')


def load_points():
    if not CONFIG_PATH.is_file():
        return {k: v.copy() for k, v in DEFAULT_POINTS.items()}
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
        points = data['points']
        for key in KEYS:
            p = points[key]
            if not isinstance(p, list) or len(p) != 2 or not all(isinstance(n, (int, float)) and 0 <= n <= 1 for n in p):
                raise ValueError(f'Ponto invalido: {key}')
        return {k: points[k].copy() for k in KEYS}
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f'[CONFIG] Configuracao invalida ({exc}). Usando padrao.')
        return {k: v.copy() for k, v in DEFAULT_POINTS.items()}


def save_points(points):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {'reference': 'normalized', 'points': points}
    CONFIG_PATH.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f'[CONFIG] Salvo em {CONFIG_PATH}')


def pixel_points(points, w, h):
    return {k: (round(v[0] * w), round(v[1] * h)) for k, v in points.items()}


def signed_side(a, b, p):
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def valid_gate(points, w, h):
    p = pixel_points(points, w, h)
    il, ir, ol, or_ = (p[k] for k in KEYS)
    if not (il[0] < ir[0] and ol[0] < or_[0]):
        return False
    if not (max(il[1], ir[1]) + 20 < min(ol[1], or_[1])):
        return False
    polygon = np.array([il, ir, or_, ol], dtype=np.int32)
    return cv2.contourArea(polygon) > 500


def point_zone(x, y, points, w, h):
    p = pixel_points(points, w, h)
    il, ir, ol, or_ = (p[k] for k in KEYS)
    # A largura do corredor varia linearmente com a profundidade.
    mid_inner_y = (il[1] + ir[1]) / 2
    mid_outer_y = (ol[1] + or_[1]) / 2
    t = (y - mid_inner_y) / max(1, mid_outer_y - mid_inner_y)
    left_x = il[0] + t * (ol[0] - il[0])
    right_x = ir[0] + t * (or_[0] - ir[0])
    if not (left_x <= x <= right_x):
        return 'OUTSIDE'
    # Segmentos inclinados: usa o lado relativo de cada linha.
    inner_side = signed_side(il, ir, (x, y))
    outer_side = signed_side(ol, or_, (x, y))
    margin = 8 * w / REF_W
    inner_len = max(1, ((ir[0]-il[0])**2 + (ir[1]-il[1])**2)**0.5)
    outer_len = max(1, ((or_[0]-ol[0])**2 + (or_[1]-ol[1])**2)**0.5)
    if inner_side / inner_len < -margin:
        return 'INSIDE'
    if outer_side / outer_len > margin:
        return 'SIDEWALK'
    return 'CROSSING'


class GateCounter:
    def __init__(self):
        self.entries = 0
        self.exits = 0
        self.states = {}
        self.events = deque(maxlen=5)

    def update(self, tracked, shape, timestamp, points):
        h, w = shape[:2]
        for track_id in list(self.states):
            if timestamp - self.states[track_id]['last_seen'] > TRACK_TIMEOUT_SECONDS:
                del self.states[track_id]
        if tracked.tracker_id is None:
            return
        for i in range(len(tracked)):
            tid = int(tracked.tracker_id[i])
            if tid < 0:
                continue
            x1, y1, x2, y2 = map(int, tracked.xyxy[i])
            zone = point_zone((x1+x2)//2, y2, points, w, h)
            state = self.states.setdefault(tid, {'origin': None, 'origin_time': None, 'last_seen': timestamp})
            state['last_seen'] = timestamp
            if zone == 'OUTSIDE':
                continue
            if state['origin'] is None:
                if zone in ('INSIDE', 'SIDEWALK'):
                    state['origin'], state['origin_time'] = zone, timestamp
                continue
            if timestamp - state['origin_time'] > MAX_CROSSING_SECONDS:
                state['origin'] = zone if zone in ('INSIDE', 'SIDEWALK') else None
                state['origin_time'] = timestamp if state['origin'] else None
                continue
            if state['origin'] == 'SIDEWALK' and zone == 'INSIDE':
                self.entries += 1
                self.events.appendleft(f'IN | ID {tid}')
                print(f'IN | Pessoa #{tid}')
                state['origin'], state['origin_time'] = 'INSIDE', timestamp
            elif state['origin'] == 'INSIDE' and zone == 'SIDEWALK':
                self.exits += 1
                self.events.appendleft(f'OUT | ID {tid}')
                print(f'OUT | Pessoa #{tid}')
                state['origin'], state['origin_time'] = 'SIDEWALK', timestamp


class CalibrationUI:
    def __init__(self, points):
        self.points = points
        self.active = False
        self.dragged = None
        self.display_w = 1
        self.display_h = 1
        self.video_w = 1
        self.video_h = 1
        self.dirty = False

    def set_size(self, video_w, video_h, display_w, display_h):
        self.video_w, self.video_h = video_w, video_h
        self.display_w, self.display_h = display_w, display_h

    def on_mouse(self, event, x, y, flags, userdata):
        if not self.active:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            nearest, dist = None, float('inf')
            for k, p in self.points.items():
                px, py = p[0] * self.display_w, p[1] * self.display_h
                d = ((x-px)**2 + (y-py)**2)**0.5
                if d < dist:
                    nearest, dist = k, d
            if dist <= HANDLE_RADIUS * 2:
                self.dragged = nearest
        elif event == cv2.EVENT_MOUSEMOVE and self.dragged is not None:
            candidate = {k: v.copy() for k, v in self.points.items()}
            candidate[self.dragged] = [max(0, min(1, x / self.display_w)), max(0, min(1, y / self.display_h))]
            if valid_gate(candidate, self.video_w, self.video_h):
                self.points = candidate
                self.dirty = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragged = None


# Visual theme: OpenCV uses BGR colors.
BG = (30, 27, 23)
PANEL = (48, 43, 37)
ACCENT = (135, 221, 102)
CYAN = (235, 202, 92)
MUTED = (185, 180, 170)
CORAL = (115, 132, 250)


def rounded_rect(img, p1, p2, radius, color, thickness=-1):
    x1, y1 = p1
    x2, y2 = p2
    if x2 <= x1 or y2 <= y1:
        return
    radius = max(0, min(radius, (x2-x1)//2, (y2-y1)//2))
    if thickness < 0:
        cv2.rectangle(img, (x1+radius, y1), (x2-radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1+radius), (x2, y2-radius), color, -1)
        for cx, cy in ((x1+radius,y1+radius),(x2-radius,y1+radius),
                       (x1+radius,y2-radius),(x2-radius,y2-radius)):
            cv2.circle(img, (cx,cy), radius, color, -1, cv2.LINE_AA)
    else:
        cv2.line(img, (x1+radius,y1),(x2-radius,y1),color,thickness,cv2.LINE_AA)
        cv2.line(img, (x1+radius,y2),(x2-radius,y2),color,thickness,cv2.LINE_AA)
        cv2.line(img, (x1,y1+radius),(x1,y2-radius),color,thickness,cv2.LINE_AA)
        cv2.line(img, (x2,y1+radius),(x2,y2-radius),color,thickness,cv2.LINE_AA)
        for cx,cy,start,end in ((x1+radius,y1+radius,180,270),(x2-radius,y1+radius,270,360),
                                (x1+radius,y2-radius,90,180),(x2-radius,y2-radius,0,90)):
            cv2.ellipse(img,(cx,cy),(radius,radius),0,start,end,color,thickness,cv2.LINE_AA)


def translucent_panel(frame, p1, p2, color=BG, opacity=.86, radius=14):
    layer = frame.copy()
    rounded_rect(layer, p1, p2, radius, color)
    cv2.addWeighted(layer, opacity, frame, 1-opacity, 0, frame)


def text(frame, value, pos, scale=.5, color=WHITE, thickness=1):
    cv2.putText(frame, str(value), pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def draw_overlay(frame, points, ui, counter, video_fps, ai_fps, paused):
    h, w = frame.shape[:2]
    p = pixel_points(points, w, h)
    il, ir, ol, or_ = (p[k] for k in KEYS)
    poly = np.array([il, ir, or_, ol], dtype=np.int32)

    # Subtle monitored corridor with clean boundary lines.
    layer = frame.copy()
    cv2.fillPoly(layer, [poly], ACCENT)
    cv2.addWeighted(layer, .085, frame, .915, 0, frame)
    cv2.line(frame, il, ir, (75, 217, 252), 3, cv2.LINE_AA)
    cv2.line(frame, ol, or_, CYAN, 3, cv2.LINE_AA)
    cv2.line(frame, il, ol, ACCENT, 1, cv2.LINE_AA)
    cv2.line(frame, ir, or_, ACCENT, 1, cv2.LINE_AA)

    if ui.active:
        for k, pos in p.items():
            color = (75, 217, 252) if k.startswith('inner') else CYAN
            cv2.circle(frame, pos, 13, BG, -1, cv2.LINE_AA)
            cv2.circle(frame, pos, 9, color, -1, cv2.LINE_AA)
            cv2.circle(frame, pos, 4, WHITE, -1, cv2.LINE_AA)
            text(frame, k.replace('_', ' ').upper(), (pos[0]+15, pos[1]-14), .4, WHITE, 1)

    # Compact top dashboard, responsive to the frame width.
    pad = max(12, round(w*.014))
    top_h = 118 if w >= 850 else 102
    translucent_panel(frame, (pad, pad), (w-pad, top_h), BG, .88, 16)
    text(frame, 'REALI', (pad+18, pad+26), .68 if w>=850 else .53, WHITE, 2)
    status = 'CALIBRANDO' if ui.active else ('PAUSADO' if paused else 'MONITORANDO')
    status_color = (75,217,252) if ui.active else (75,217,252) if paused else ACCENT
    sx = w-pad-205 if w >= 850 else w-pad-125
    if sx > pad+205:
        cv2.circle(frame, (sx, pad+20), 5, status_color, -1, cv2.LINE_AA)
        text(frame, status, (sx+12, pad+25), .44 if w>=850 else .35, WHITE, 1)

    metric_y = pad + (72 if w>=850 else 66)
    metric_x = [pad+20, pad+max(145,w//4), pad+max(290,w//2)]
    for x, label, value, color in (
        (metric_x[0], 'ENTRADAS', counter.entries, ACCENT),
        (metric_x[1], 'SAIDAS', counter.exits, CORAL),
        (metric_x[2], 'SALDO', counter.entries-counter.exits, WHITE)
    ):
        text(frame, label, (x, metric_y-14), .38, MUTED)
        text(frame, value, (x, metric_y+14), .82 if w>=850 else .66, color, 2)

    # Small bottom info bar; avoid covering most of the scene.
    footer_y = h-47
    translucent_panel(frame, (pad, footer_y), (w-pad, h-pad), BG, .83, 10)
    text(frame, f'VIDEO {video_fps:.1f} FPS   |   IA {ai_fps:.1f} FPS',
         (pad+12, footer_y+22), .43 if w>=850 else .34, WHITE)
    if w >= 850:
        text(frame, 'C CALIBRAR   S SALVAR   E EXPORTAR   R REINICIAR   P PAUSAR   Q SAIR',
             (w-pad-490, footer_y+22), .38, MUTED)

    # Recent events: discrete chips on the right, away from the counting zone.
    event_y = top_h+24
    for event in list(counter.events)[:3]:
        chip_w = 160
        x1 = w-pad-chip_w
        if event_y+30 >= footer_y:
            break
        translucent_panel(frame, (x1,event_y), (w-pad,event_y+29), BG, .83, 8)
        color = ACCENT if event.startswith('IN') else CORAL
        cv2.circle(frame, (x1+13,event_y+15), 4, color, -1, cv2.LINE_AA)
        text(frame, event, (x1+25,event_y+20), .42, WHITE)
        event_y += 36


def draw_tracks(frame, tracked, points):
    if tracked is None or tracked.tracker_id is None:
        return
    h, w = frame.shape[:2]
    for i in range(len(tracked)):
        tid = int(tracked.tracker_id[i])
        if tid < 0:
            continue
        x1, y1, x2, y2 = map(int, tracked.xyxy[i])
        x1, x2 = max(0,x1), min(w-1,x2)
        y1, y2 = max(0,y1), min(h-1,y2)
        if x2 <= x1 or y2 <= y1:
            continue
        zone = point_zone((x1+x2)//2, y2, points, w, h)
        active = zone != 'OUTSIDE'
        color = ACCENT if active else CYAN
        thickness = max(2, round(w/650))

        # Soft halo plus rounded corners; cleaner than a solid rectangle.
        cv2.rectangle(frame, (x1,y1), (x2,y2), BG, thickness+3, cv2.LINE_AA)
        rounded_rect(frame, (x1,y1), (x2,y2), min(12,(x2-x1)//7), color, thickness)

        # ID badge pinned to the upper edge of the detection box.
        label = f'PERSON  {tid:03d}'
        (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .43, 1)
        badge_w = tw+20
        badge_x = max(0, min(x1, w-badge_w-1))
        badge_y = max(0, y1-27)
        rounded_rect(frame, (badge_x,badge_y), (badge_x+badge_w,badge_y+24), 6, BG)
        cv2.rectangle(frame, (badge_x,badge_y+21), (badge_x+badge_w,badge_y+24), color, -1)
        text(frame, label, (badge_x+10,badge_y+17), .43, WHITE)

        # Small anchor at estimated feet, used by the counter.
        foot = ((x1+x2)//2,y2)
        cv2.circle(frame, foot, 7, BG, -1, cv2.LINE_AA)
        cv2.circle(frame, foot, 4, color, -1, cv2.LINE_AA)

def export_video(detector, points, fps_video, expected_frames=0):
    """Reprocessa o vídeo inteiro e salva a visualização anotada em MP4."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"reali_vision_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.mp4"
    output_path = OUTPUT_DIR / filename

    source = cv2.VideoCapture(str(VIDEO_PATH))
    if not source.isOpened():
        print('[EXPORT] Não foi possível abrir o vídeo de origem.')
        return

    writer = None
    frames_done = 0
    finished = False
    tracker = PersonTracker()
    counter = GateCounter()
    durations = deque(maxlen=60)
    start_export = time.perf_counter()

    try:
        width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise RuntimeError('Resolução do vídeo inválida')

        # MP4V funciona com instalações comuns do OpenCV no Windows.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps_video, (width, height))
        if not writer.isOpened():
            raise RuntimeError('Não foi possível iniciar o VideoWriter MP4')

        print(f'\n[EXPORT] Salvando: {output_path}')
        print('[EXPORT] Processando todos os frames; pressione Q para cancelar.')
        while True:
            start_frame = time.perf_counter()
            ok, frame = source.read()
            if not ok:
                finished = True
                break

            timestamp = frames_done / fps_video
            detections = detector.detect(frame)
            tracked = tracker.update(detections, timestamp=timestamp)
            counter.update(tracked, frame.shape, timestamp, points)
            durations.append(time.perf_counter() - start_frame)
            ai_fps = len(durations) / sum(durations) if durations else 0.0

            # Renderiza no tamanho ORIGINAL para exportar com melhor qualidade.
            draw_tracks(frame, tracked, points)
            export_ui = type('ExportUI', (), {'active': False})()
            draw_overlay(frame, points, export_ui, counter, fps_video, ai_fps, False)
            writer.write(frame)
            frames_done += 1

            if frames_done % 60 == 0:
                percent = (100 * frames_done / expected_frames) if expected_frames else 0
                print(f'[EXPORT] {frames_done} frames | {percent:.1f}% | IN={counter.entries} OUT={counter.exits}')

            # Mantém a janela responsiva durante a exportação.
            preview_width = min(width, DISPLAY_WIDTH)
            preview_height = round(height * preview_width / width)
            preview = cv2.resize(frame, (preview_width, preview_height)) if preview_width != width else frame
            cv2.imshow(WINDOW, preview)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('[EXPORT] Cancelado pelo usuário.')
                break

    except Exception as exc:
        print(f'[EXPORT] Erro: {exc}')
    finally:
        source.release()
        if writer is not None:
            writer.release()
            converter_para_h264(output_path)

    if finished and frames_done:
        print(f'[EXPORT] Concluído: {output_path}')
        print(f'[EXPORT] {frames_done} frames | IN={counter.entries} OUT={counter.exits}')
        print(f'[EXPORT] Tempo de processamento: {time.perf_counter()-start_export:.1f}s')
    else:
        # Não deixa arquivos incompletos confundirem os resultados.
        if output_path.exists():
            try:
                output_path.unlink()
            except OSError:
                print(f'[EXPORT] Arquivo parcial: {output_path}')


def converter_para_h264(video_path):
    video_path = Path(video_path)

    saida = video_path.with_name(
        video_path.stem + "_h264.mp4"
    )

    try:
        subprocess.run([
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(saida)
        ], check=True)

        print(f"[REALI] Vídeo convertido: {saida}")
        return saida

    except FileNotFoundError:
        print("[ERRO] FFmpeg não encontrado.")
        return None

    except subprocess.CalledProcessError:
        print("[ERRO] Falha na conversão.")
        return None

def main():
    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(f'Video nao encontrado: {VIDEO_PATH}')
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f'Modelo nao encontrado: {MODEL_PATH}')
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f'Falha ao abrir {VIDEO_PATH}')
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    if fps_video <= 0 or fps_video > 240:
        fps_video = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    display_w = min(w, DISPLAY_WIDTH)
    display_h = round(h * display_w / w)
    ui = CalibrationUI(load_points())
    ui.set_size(w, h, display_w, display_h)
    detector = PersonDetector(model_path=str(MODEL_PATH), confidence=CONFIDENCE)
    def reset():
        return PersonTracker(), GateCounter(), deque(maxlen=60)
    tracker, counter, processing_times = reset()
    frame_index = 0
    paused = False
    ai_fps = 0.0
    last_frame = None
    last_tracked = None
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, ui.on_mouse)
    print('C: calibrar | Arraste pontos | S: salvar | E: exportar MP4 | R: reiniciar | P: pausar | Q: sair')
    print('Ao sair da calibracao, o video reinicia para evitar contagens inconsistentes.')
    try:
        while True:
            start = time.perf_counter()
            if not paused and not ui.active:
                ok, frame = cap.read()
                if not ok:
                    print('Fim do video.')
                    break
                frame_index += 1
                timestamp = (frame_index - 1) / fps_video
                detections = detector.detect(frame)
                tracked = tracker.update(detections, timestamp=timestamp)
                counter.update(tracked, frame.shape, timestamp, ui.points)
                elapsed = time.perf_counter() - start
                processing_times.append(elapsed)
                ai_fps = 1 / (sum(processing_times) / len(processing_times))
                last_frame = frame
                last_tracked = tracked
            if last_frame is not None:
                display = last_frame.copy()
                if not ui.active:
                    draw_tracks(display, last_tracked, ui.points)
                draw_overlay(display, ui.points, ui, counter, fps_video, ai_fps, paused)
                if display_w != w:
                    display = cv2.resize(display, (display_w, display_h), interpolation=cv2.INTER_AREA)
                cv2.imshow(WINDOW, display)
            elapsed_total = time.perf_counter() - start
            delay = 30 if (paused or ui.active) else max(1, round(1000 * max(0, 1 / fps_video - elapsed_total)))
            key = cv2.waitKey(delay) & 0xFF
            if key == ord('q'):
                if ui.dirty:
                    print('[CONFIG] Ha alteracoes nao salvas. Pressione S antes de sair na proxima vez.')
                break
            if key == ord('c'):
                ui.active = not ui.active
                ui.dragged = None
                paused = False
                if ui.active:
                    if last_frame is None:
                        ok, last_frame = cap.read()
                        if not ok:
                            break
                    print('[CALIBRACAO] Arraste os quatro pontos. S para salvar.')
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    tracker, counter, processing_times = reset()
                    frame_index, ai_fps, last_tracked, last_frame = 0, 0.0, None, None
                    print('[CALIBRACAO] Video e contagem reiniciados.')
            elif key == ord('e'):
                if ui.active:
                    print('[EXPORT] Saia da calibração com C antes de exportar.')
                else:
                    if ui.dirty:
                        print('[EXPORT] Aviso: coordenadas ainda não salvas. Usando as posições atuais.')
                    export_video(detector, {k: v.copy() for k, v in ui.points.items()}, fps_video, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
                    print('[REALI] Exportação encerrada; reprodução preservada.')
            elif key == ord('s'):
                save_points(ui.points)
                ui.dirty = False
            elif key == ord('p') and not ui.active:
                paused = not paused
            elif key == ord('r'):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                tracker, counter, processing_times = reset()
                frame_index, ai_fps, last_tracked, last_frame = 0, 0.0, None, None
                paused = False
                ui.active = False
                print('[REALI] Reiniciado.')
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f'Resultado: IN={counter.entries} OUT={counter.exits} SALDO={counter.entries-counter.exits}')


if __name__ == '__main__':
    main()
