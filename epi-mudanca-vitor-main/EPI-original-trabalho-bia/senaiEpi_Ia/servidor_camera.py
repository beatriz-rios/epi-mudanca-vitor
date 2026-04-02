import cv2
from ultralytics import YOLO
import threading
import time

# ==============================================================================
# 1. CONFIGURAÇÕES DA IA
# ==============================================================================
# O caminho exato da SUA IA que você forneceu
CAMINHO_MODELO_CUSTOM = "C:\\xampp\\htdocs\\epi-mudanca-vitor\\epi-mudanca-vitor-main\\EPI-original-trabalho-bia\\senaiEpi_Ia\\best (2) (1).pt" 

# ==============================================================================
# VARIÁVEIS GLOBAIS
# ==============================================================================
frame_atual = None
lock_frame = threading.Lock()
caixas_detectadas = []

# ==============================================================================
# 2. INICIALIZAÇÃO DA SUA IA
# ==============================================================================
print(f"[SISTEMA] Carregando a SUA IA ({CAMINHO_MODELO_CUSTOM})...")
try:
    model_custom = YOLO(CAMINHO_MODELO_CUSTOM)
    print("[SISTEMA] IA Customizada carregada com sucesso!")
except Exception as e:
    print(f"[ERRO CRÍTICO] Falha ao carregar {CAMINHO_MODELO_CUSTOM}. Erro: {e}")
    exit()

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
        else: 
            time.sleep(0.1)

def processar_ia():
    global frame_atual, caixas_detectadas

    while True:
        if frame_atual is None:
            time.sleep(0.01)
            continue

        with lock_frame:
            img_process = frame_atual.copy()

        # Roda APENAS a sua IA
        # conf=0.25 significa que ela mostrará a caixa se tiver 25% de certeza ou mais
        results_custom = model_custom.predict(img_process, conf=0.25, verbose=False)
        
        temp_caixas = []
        
        # Pegando os resultados da IA (coordenadas, confiança e nome)
        for r in results_custom:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confianca = float(box.conf[0]) * 100
                classe_id = int(box.cls[0])
                nome_classe = model_custom.names[classe_id] # Pega o nome que você colocou no Roboflow
                
                texto = f"{nome_classe} {confianca:.1f}%"
                temp_caixas.append((x1, y1, x2, y2, texto))

        caixas_detectadas = temp_caixas
        time.sleep(0.01)

# ==============================================================================
# 4. EXIBIÇÃO LOCAL
# ==============================================================================
def mostrar_na_janela():
    while True:
        if frame_atual is not None:
            with lock_frame:
                vis_frame = frame_atual.copy()

            # Desenha os retângulos encontrados pela SUA IA
            for (x1, y1, x2, y2, txt) in caixas_detectadas:
                cor = (255, 0, 255) # Cor Rosa/Magenta para destacar bem
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), cor, 2) 
                cv2.putText(vis_frame, txt, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)

            cv2.imshow("TESTE EXCLUSIVO - SUA IA (best.pt)", vis_frame)
            
            # Aperte 'q' ou 'ESC' no teclado para sair
            if cv2.waitKey(1) & 0xFF in [27, ord('q')]: 
                break
        else: 
            time.sleep(0.1)
            
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print("="*60)
    print("INICIANDO SISTEMA SOMENTE COM A SUA IA")
    print("="*60)
    
    threading.Thread(target=capturar_frames, daemon=True).start()
    threading.Thread(target=processar_ia, daemon=True).start()
    
    mostrar_na_janela()