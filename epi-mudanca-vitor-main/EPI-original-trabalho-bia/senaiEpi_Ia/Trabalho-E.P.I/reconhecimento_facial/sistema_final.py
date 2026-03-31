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

print("[SISTEMA] Carregando a IA Especialista...")
# Carrega a IA que você treinou
CAMINHO_MODELO = r"c:\xampp\htdocs\epi-mudanca-vitor\epi-mudanca-vitor-main\EPI-original-trabalho-bia\senaiEpi_Ia\best.pt"
model = YOLO(CAMINHO_MODELO)

# ==============================================================================
# 2. FUNÇÃO DE VALIDAÇÃO DE CORES (HSV) - O "TIRA-TEIMA"
# ==============================================================================
def verificar_hsv_capacete(img_crop):
    """
    Verifica se a área recortada pela IA realmente possui as cores do EPI.
    """
    if img_crop is None or img_crop.size == 0:
        return False, 0.0, 0.0
    h, w = img_crop.shape[:2]
    
    hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
    
    # Azul escuro (aba e detalhe central)
    lower_blue = np.array([80, 40, 20])
    upper_blue = np.array([150, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Preto (telas laterais)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 80])
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    
    kernel = np.ones((3, 3), np.uint8)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
    mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_OPEN, kernel)
    
    area_total = h * w
    ratio_blue = cv2.countNonZero(mask_blue) / area_total
    ratio_black = cv2.countNonZero(mask_black) / area_total
    
    # Exige no mínimo 2% de azul e 5% de preto na área recortada
    tem_azul = ratio_blue >= 0.02   
    tem_preto = ratio_black >= 0.05 
    
    return (tem_azul and tem_preto), ratio_blue, ratio_black

# ==============================================================================
# 3. THREADS DE VÍDEO E IA
# ==============================================================================
def capturar_frames():
    global frame_atual, camera_ativa
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, 1280)
    cap.set(4, 720)
    
    print("[SISTEMA] Webcam iniciada.")
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

        # A IA procura possíveis bonés. 
        # Confiança baixa (0.30) para ela achar mais coisas e deixar o HSV fazer o desempate rigoroso.
        results = model.predict(frame, conf=0.30, verbose=False, imgsz=640)

        capacetes_validados = []
        
        for r in results:
            for box in r.boxes:
                hx1, hy1, hx2, hy2 = map(int, box.xyxy[0])
                
                # Corta a imagem exatamente onde a IA achou o boné
                recorte_bone = frame[hy1:hy2, hx1:hx2]
                
                # Passa o recorte pela validação de cores
                valido, r_blue, r_black = verificar_hsv_capacete(recorte_bone)
                
                if valido:
                    status = f"EPI OK! Az:{r_blue:.1%} Pr:{r_black:.1%}"
                    cor = (0, 255, 0) # Verde = Boné Correto
                else:
                    status = f"IGNORADO. Az:{r_blue:.1%} Pr:{r_black:.1%}"
                    cor = (0, 0, 255) # Vermelho = A IA achou, mas as cores não bateram
                    
                capacetes_validados.append((hx1, hy1, hx2, hy2, status, cor))

        ultimo_desenho_capacetes = capacetes_validados
        time.sleep(0.01)

# ==============================================================================
# 4. EXIBIÇÃO NO TERMINAL
# ==============================================================================
def exibir_janela():
    global camera_ativa

    while camera_ativa:
        if frame_atual is None:
            time.sleep(0.05)
            continue

        with lock_frame:
            frame_display = frame_atual.copy()

        # Desenha os resultados na tela
        for hx1, hy1, hx2, hy2, status, cor in ultimo_desenho_capacetes:
            cv2.rectangle(frame_display, (hx1, hy1), (hx2, hy2), cor, 2) 
            cv2.putText(frame_display, status, (hx1, hy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)

        cv2.imshow("EPI Guard - IA + Teste de Cores", frame_display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[SISTEMA] Encerrando...")
            camera_ativa = False
            break

        if cv2.getWindowProperty("EPI Guard - IA + Teste de Cores", cv2.WND_PROP_VISIBLE) < 1:
            camera_ativa = False
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    print("\n" + "=" * 55)
    print("   EPI GUARD v4.0 - YOLO (IA) + VALIDACAO DE CORES")
    print("   Janela OpenCV: Pressione 'Q' para Sair")
    print("=" * 55 + "\n")

    threading.Thread(target=capturar_frames, daemon=True).start()
    threading.Thread(target=processar_ia, daemon=True).start()
    exibir_janela()