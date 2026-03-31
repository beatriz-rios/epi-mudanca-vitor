import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ==============================================================================
# 1. CONFIGURAÇÕES DE IA (SISTEMA HÍBRIDO)
# ==============================================================================
# Voltamos a procurar o capacete, mas usaremos a cor AZUL como plano de backup para a frente!
CLASSES_YOLO = ["person", "helmet", "hard hat"]
PERSON_CLASS = 0
HELMET_CLASSES = [1, 2]

# ==============================================================================
# VARIÁVEIS GLOBAIS
# ==============================================================================
frame_atual = None
lock_frame = threading.Lock()
ultimo_desenho_capacetes = []
foco_nome = "Usuario Teste"
foco_status = "BUSCANDO..."
foco_cor = (0, 0, 255)
foco_bbox = None

# ==============================================================================
# 2. INICIALIZAÇÃO
# ==============================================================================
print("[SISTEMA] Carregando YOLO-World (Aguarde)...")
model = YOLO("yolov8s-worldv2.pt")
model.set_classes(CLASSES_YOLO)

# ==============================================================================
# 3. VALIDAÇÃO DE COR (SOMENTE AZUL DA FAIXA FRONTAL)
# ==============================================================================
def verificar_azul_frente(img_crop):
    if img_crop is None or img_crop.size == 0:
        return False, 0.0
    
    hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
    
    # Busca estrita pela faixa azul (ignora cabelos e sombras)
    lower_blue = np.array([90, 40, 20]) 
    upper_blue = np.array([145, 255, 255])
    
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    area_total = img_crop.shape[0] * img_crop.shape[1]
    ratio_blue = cv2.countNonZero(mask_blue) / area_total
    
    # Requer apenas 3.5% da cabeça preenchida com a faixa azul
    tem_azul = ratio_blue >= 0.035 
    
    return tem_azul, ratio_blue

# ==============================================================================
# 4. PROCESSAMENTO DAS THREADS
# ==============================================================================
def capturar_frames():
    global frame_atual
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, 1280)
    cap.set(4, 720)
    while True:
        ret, frame = cap.read()
        if ret:
            with lock_frame:
                frame_atual = frame.copy()
        else: time.sleep(0.1)

def processar_ia():
    global frame_atual, ultimo_desenho_capacetes, foco_status, foco_cor, foco_bbox

    while True:
        if frame_atual is None:
            time.sleep(0.01)
            continue

        with lock_frame:
            img_process = frame_atual.copy()

        # Confiança ajustada para captar pessoas e formatos de capacete
        results = model.predict(img_process, conf=0.30, verbose=False)
        
        pessoas_yolo = []
        capacetes_yolo = []
        
        for r in results:
            for box in r.boxes:
                coords = list(map(int, box.xyxy[0]))
                cls = int(box.cls[0])
                if cls == PERSON_CLASS:
                    pessoas_yolo.append(coords)
                elif cls in HELMET_CLASSES:
                    capacetes_yolo.append(coords)

        pessoa_foco = None
        maior_area = 0
        for p in pessoas_yolo:
            area = (p[2]-p[0]) * (p[3]-p[1])
            if area > maior_area:
                maior_area = area
                pessoa_foco = p

        temp_cap_desenho = []
        
        if pessoa_foco is not None:
            foco_bbox = pessoa_foco
            px1, py1, px2, py2 = pessoa_foco
            h_pessoa = py2 - py1
            w_pessoa = px2 - px1
            
            # ------------------------------------------------------------------
            # PASSO 1: O YOLO VIU O FORMATO DO CAPACETE NAS LATERAIS?
            # ------------------------------------------------------------------
            tem_capacete_formato = False
            zona_acima_cabeca = py1 - int(h_pessoa * 0.20)
            limite_y_olhos = py1 + int(h_pessoa * 0.20)
            
            for (cx1, cy1, cx2, cy2) in capacetes_yolo:
                hcx = (cx1 + cx2) / 2
                hcy = (cy1 + cy2) / 2
                # Se o centro do capacete detectado estiver na região da cabeça desta pessoa:
                if px1 <= hcx <= px2 and zona_acima_cabeca <= hcy <= limite_y_olhos:
                    tem_capacete_formato = True
                    break

            # ------------------------------------------------------------------
            # PASSO 2: CORTAR A CABEÇA PARA LER A COR AZUL DA FRENTE
            # ------------------------------------------------------------------
            hx1 = max(0, px1 + int(w_pessoa * 0.25)) 
            hx2 = min(img_process.shape[1], px2 - int(w_pessoa * 0.25))
            hy1 = max(0, py1 - int(h_pessoa * 0.08)) 
            hy2 = min(img_process.shape[0], py1 + int(h_pessoa * 0.12))
            
            tem_azul_frente = False
            r_blue = 0.0
            
            if hx2 > hx1 and hy2 > hy1:
                img_cabeca = img_process[hy1:hy2, hx1:hx2]
                tem_azul_frente, r_blue = verificar_azul_frente(img_cabeca)
            
            # ------------------------------------------------------------------
            # DECISÃO FINAL: APROVA SE TIVER O FORMATO(LADO) OU A COR AZUL(FRENTE)
            # ------------------------------------------------------------------
            capacete_valido = tem_capacete_formato or tem_azul_frente
            
            if tem_capacete_formato and tem_azul_frente:
                txt = f"APROV: 100% (Formato+Azul:{r_blue:.1%})"
            elif tem_capacete_formato:
                txt = "APROV: YOLO (Formato Lateral)"
            elif tem_azul_frente:
                txt = f"APROV: COR (Faixa Azul:{r_blue:.1%})"
            else:
                txt = "REPROVADO: CABELO/SEM EPI"

            cor_caixa = (0, 255, 0) if capacete_valido else (0, 0, 255)
            
            # Desenha a caixa na região da cabeça
            temp_cap_desenho.append((hx1, hy1, hx2, hy2, cor_caixa, txt))
            
            if capacete_valido:
                foco_status = "EPI APROVADO"
                foco_cor = (0, 255, 0)
            else:
                foco_status = "ALERTA: SEM CAPACETE"
                foco_cor = (0, 0, 255)
        
        ultimo_desenho_capacetes = temp_cap_desenho
        time.sleep(0.01)

# ==============================================================================
# 5. EXIBIÇÃO LOCAL
# ==============================================================================
def mostrar_na_janela():
    while True:
        if frame_atual is not None:
            with lock_frame:
                vis_frame = frame_atual.copy()

            for (x1, y1, x2, y2, cor, txt) in ultimo_desenho_capacetes:
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), cor, 2) 
                # Ajuste no texto para caber bem na tela
                cv2.putText(vis_frame, txt, (x1-10, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor, 2)

            if foco_bbox is not None:
                cv2.rectangle(vis_frame, (foco_bbox[0], foco_bbox[1]), (foco_bbox[2], foco_bbox[3]), foco_cor, 2)
                cv2.putText(vis_frame, f"{foco_status}", (foco_bbox[0]+5, foco_bbox[1]+25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, foco_cor, 2)

            cv2.imshow("TESTE - ANTI FALSO-POSITIVO (HIBRIDO)", vis_frame)
            if cv2.waitKey(1) & 0xFF in [27, ord('q')]: break
        else: time.sleep(0.1)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    threading.Thread(target=capturar_frames, daemon=True).start()
    threading.Thread(target=processar_ia, daemon=True).start()
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False), daemon=True).start()
    
    print("="*50)
    print("SISTEMA HÍBRIDO ATIVADO")
    print("De Lado: Valida pelo formato do capacete (Ignora cabelo preto)")
    print("De Frente: Valida pela faixa Azul Escura")
    print("="*50)
    
    mostrar_na_janela()