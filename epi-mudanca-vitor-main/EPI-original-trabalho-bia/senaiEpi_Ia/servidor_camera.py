import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time

# ==============================================================================
# 1. CONFIGURAÇÕES GERAIS
# ==============================================================================
camera_ativa = True
frame_atual = None
lock_frame = threading.Lock()
ultimo_desenho_capacetes = []

# --- AJUSTE O CAMINHO DO SEU MODELO AQUI ---
CAMINHO_MODELO = r"C:\xampp\htdocs\epi-mudanca-vitor\epi-mudanca-vitor-main\EPI-original-trabalho-bia\senaiEpi_Ia\Trabalho-E.P.I\reconhecimento_facial\best (2).pt"

print(f"[SISTEMA] Carregando IA: {CAMINHO_MODELO}")
model = YOLO(CAMINHO_MODELO)

# ==============================================================================
# 2. VALIDAÇÃO DE CORES (Ajustada para ser mais permissiva)
# ==============================================================================
def verificar_hsv_capacete(img_crop):
    if img_crop is None or img_crop.size == 0:
        return False, 0.0, 0.0
    
    h, w = img_crop.shape[:2]
    hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
    
    # Azul: Ajustado para pegar desde o azul marinho até o azul brilhante
    lower_blue = np.array([85, 40, 20]) 
    upper_blue = np.array([135, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Preto: Focado em tons bem escuros
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 60])
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    
    # Limpeza de ruído
    kernel = np.ones((3, 3), np.uint8)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
    
    area_total = h * w
    ratio_blue = cv2.countNonZero(mask_blue) / area_total
    ratio_black = cv2.countNonZero(mask_black) / area_total
    
    # CRITÉRIO: Pelo menos 1.5% de azul (ajustado de 2%) e 3% de preto (ajustado de 5%)
    tem_azul = ratio_blue >= 0.015
    tem_preto = ratio_black >= 0.03
    
    return (tem_azul and tem_preto), ratio_blue, ratio_black

# ==============================================================================
# 3. PROCESSAMENTO (IA + LÓGICA DE DIAGNÓSTICO)
# ==============================================================================
def capturar_frames():
    global frame_atual, camera_ativa
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, 1280)
    cap.set(4, 720)
    
    while camera_ativa:
        ret, frame = cap.read()
        if ret:
            with lock_frame:
                frame_atual = frame.copy()
        else:
            time.sleep(0.01)
    cap.release()

def processar_ia():
    global frame_atual, camera_ativa, ultimo_desenho_capacetes

    while camera_ativa:
        if frame_atual is None:
            time.sleep(0.01)
            continue

        with lock_frame:
            frame = frame_atual.copy()

        # --- ALTERAÇÃO: conf=0.25 para ser mais sensível ---
        results = model.predict(frame, conf=0.25, verbose=False, imgsz=640)

        capacetes_validados = []
        
        for r in results:
            for box in r.boxes:
                hx1, hy1, hx2, hy2 = map(int, box.xyxy[0])
                conf_valor = float(box.conf[0])
                
                # Recorte para análise de cor
                recorte = frame[hy1:hy2, hx1:hx2]
                valido, r_blue, r_black = verificar_hsv_capacete(recorte)
                
                # Log de Debug no console
                print(f"[DEBUG] IA: {conf_valor:.2f} | Azul: {r_blue:.1%} | Preto: {r_black:.1%}")

                if valido:
                    status = f"EPI OK ({conf_valor:.2f})"
                    cor = (0, 255, 0) # Verde
                else:
                    status = f"COR INVALIDA (Az:{r_blue:.1%})"
                    cor = (0, 255, 255) # AMARELO: Indica que a IA achou, mas a cor falhou
                    
                capacetes_validados.append((hx1, hy1, hx2, hy2, status, cor))

        ultimo_desenho_capacetes = capacetes_validados
        time.sleep(0.02) # Leve delay para não sobrecarregar CPU

# ==============================================================================
# 4. INTERFACE E EXIBIÇÃO
# ==============================================================================
def exibir_janela():
    global camera_ativa

    cv2.namedWindow("EPI GUARD v4.5 - DEBUG MODE", cv2.WINDOW_NORMAL)

    while camera_ativa:
        if frame_atual is None:
            continue

        with lock_frame:
            frame_display = frame_atual.copy()

        # Desenha as caixas detectadas
        for hx1, hy1, hx2, hy2, status, cor in ultimo_desenho_capacetes:
            cv2.rectangle(frame_display, (hx1, hy1), (hx2, hy2), cor, 3)
            # Fundo do texto para leitura
            cv2.rectangle(frame_display, (hx1, hy1-25), (hx1+220, hy1), cor, -1)
            cv2.putText(frame_display, status, (hx1+5, hy1-7), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

        cv2.imshow("EPI GUARD v4.5 - DEBUG MODE", frame_display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or cv2.getWindowProperty("EPI GUARD v4.5 - DEBUG MODE", cv2.WND_PROP_VISIBLE) < 1:
            camera_ativa = False

    cv2.destroyAllWindows()

if __name__ == '__main__':
    print("\n" + "="*50)
    print(" INICIANDO EPI GUARD - MODO DIAGNÓSTICO")
    print(" CAIXA VERDE: IA e Cor OK")
    print(" CAIXA AMARELA: IA detectou, mas cor foi rejeitada")
    print("="*50 + "\n")

    threading.Thread(target=capturar_frames, daemon=True).start()
    threading.Thread(target=processar_ia, daemon=True).start()
    exibir_janela()