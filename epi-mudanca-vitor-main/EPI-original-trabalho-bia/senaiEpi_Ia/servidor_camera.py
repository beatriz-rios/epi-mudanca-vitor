import cv2
from ultralytics import YOLO
import threading
import time
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ==============================================================================
# 1. CONFIGURAÇÕES DE IA (SISTEMA DUPLO: YOLO-World + best(3).pt)
# ==============================================================================
# Coloque aqui o caminho completo se o ficheiro não estiver na mesma pasta
CAMINHO_MODELO_CUSTOM = "C:\\xampp\\htdocs\\epi-mudanca-vitor\\epi-mudanca-vitor-main\\EPI-original-trabalho-bia\\senaiEpi_Ia\\best.pt" 

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
# 2. INICIALIZAÇÃO DAS DUAS IAs
# ==============================================================================
print("[SISTEMA] 1/2 Carregando YOLO-World (Pessoas e Capacetes Genéricos)...")
model_world = YOLO("yolov8s-worldv2.pt")
model_world.set_classes(CLASSES_YOLO)

print(f"[SISTEMA] 2/2 Carregando IA CUSTOMIZADA ({CAMINHO_MODELO_CUSTOM})...")
try:
    model_custom = YOLO(CAMINHO_MODELO_CUSTOM)
    print("[SISTEMA] IA Customizada carregada com sucesso!")
except Exception as e:
    print(f"[ERRO CRÍTICO] Falha ao carregar {CAMINHO_MODELO_CUSTOM}. Erro: {e}")

# ==============================================================================
# 3. PROCESSAMENTO DAS THREADS
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

        # Roda as DUAS IAs no mesmo frame
        results_world = model_world.predict(img_process, conf=0.30, verbose=False)
        results_custom = model_custom.predict(img_process, conf=0.25, verbose=False)
        
        pessoas_yolo = []
        capacetes_yolo_world = []
        capacetes_ia_custom = []
        
        # Lendo os resultados do YOLO-World
        for r in results_world:
            for box in r.boxes:
                coords = list(map(int, box.xyxy[0]))
                cls = int(box.cls[0])
                if cls == PERSON_CLASS:
                    pessoas_yolo.append(coords)
                elif cls in HELMET_CLASSES:
                    capacetes_yolo_world.append(coords)

        # Lendo os resultados da SUA IA (best (3).pt)
        for r in results_custom:
            for box in r.boxes:
                capacetes_ia_custom.append(list(map(int, box.xyxy[0])))

        # Encontra a pessoa principal (a maior na câmara)
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
            
            # Define a área onde o capacete deve estar (em cima da cabeça)
            zona_acima_cabeca = py1 - int(h_pessoa * 0.20)
            limite_y_olhos = py1 + int(h_pessoa * 0.20)
            
            # ------------------------------------------------------------------
            # CHECAGEM 1: SUA IA (A Especialista)
            # ------------------------------------------------------------------
            tem_capacete_custom = False
            for (cx1, cy1, cx2, cy2) in capacetes_ia_custom:
                hcx, hcy = (cx1 + cx2) / 2, (cy1 + cy2) / 2
                if px1 <= hcx <= px2 and zona_acima_cabeca <= hcy <= limite_y_olhos:
                    tem_capacete_custom = True
                    break

            # ------------------------------------------------------------------
            # CHECAGEM 2: YOLO-WORLD (O Genérico)
            # ------------------------------------------------------------------
            tem_capacete_world = False
            for (cx1, cy1, cx2, cy2) in capacetes_yolo_world:
                hcx, hcy = (cx1 + cx2) / 2, (cy1 + cy2) / 2
                if px1 <= hcx <= px2 and zona_acima_cabeca <= hcy <= limite_y_olhos:
                    tem_capacete_world = True
                    break
            
            # ------------------------------------------------------------------
            # DECISÃO FINAL: BASTA UMA DAS IAs CONFIRMAR
            # ------------------------------------------------------------------
            capacete_valido = tem_capacete_custom or tem_capacete_world
            
            if tem_capacete_custom and tem_capacete_world:
                txt = "APROV: 100% (Ambas as IAs)"
            elif tem_capacete_custom:
                txt = "APROV: SUA IA (best.pt)"
            elif tem_capacete_world:
                txt = "APROV: YOLO-World"
            else:
                txt = "REPROVADO: SEM EPI"

            # Desenha a caixa na região da cabeça para mostrar o resultado
            hx1 = max(0, px1 + int(w_pessoa * 0.20)) 
            hx2 = min(img_process.shape[1], px2 - int(w_pessoa * 0.20))
            hy1 = max(0, py1 - int(h_pessoa * 0.15)) 
            hy2 = min(img_process.shape[0], py1 + int(h_pessoa * 0.15))

            cor_caixa = (0, 255, 0) if capacete_valido else (0, 0, 255)
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
# 4. EXIBIÇÃO LOCAL
# ==============================================================================
def mostrar_na_janela():
    while True:
        if frame_atual is not None:
            with lock_frame:
                vis_frame = frame_atual.copy()

            for (x1, y1, x2, y2, cor, txt) in ultimo_desenho_capacetes:
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), cor, 2) 
                cv2.putText(vis_frame, txt, (x1-10, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor, 2)

            if foco_bbox is not None:
                cv2.rectangle(vis_frame, (foco_bbox[0], foco_bbox[1]), (foco_bbox[2], foco_bbox[3]), foco_cor, 2)
                cv2.putText(vis_frame, f"{foco_status}", (foco_bbox[0]+5, foco_bbox[1]+25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, foco_cor, 2)

            cv2.imshow("DUPLO CHECK (YoloWorld + SuaIA)", vis_frame)
            if cv2.waitKey(1) & 0xFF in [27, ord('q')]: break
        else: time.sleep(0.1)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    threading.Thread(target=capturar_frames, daemon=True).start()
    threading.Thread(target=processar_ia, daemon=True).start()
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False), daemon=True).start()
    
    print("="*60)
    print("SISTEMA DE DUPLA SEGURANÇA ATIVADO")
    print(" 1. YOLO-World (Humanos/Capacetes Genéricos)")
    print(" 2. IA Customizada 'best(3).pt' (Especialista em Capacetes)")
    print("="*60)
    
    mostrar_na_janela()