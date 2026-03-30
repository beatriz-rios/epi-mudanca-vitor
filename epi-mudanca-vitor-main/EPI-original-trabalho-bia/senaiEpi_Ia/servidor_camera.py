import cv2
import mysql.connector
import numpy as np
from ultralytics import YOLO
import threading
import time
import os
import winsound

# ==============================================================================
# 1. CONFIGURAÇÕES GERAIS E BANCO DE DADOS (epi_guard)
# ==============================================================================
DB_CONFIG = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '',
    'database': 'epiguard',
    'port': 3306
}

PASTA_EVIDENCIAS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evidencias")
os.makedirs(PASTA_EVIDENCIAS, exist_ok=True)

# Agora só usamos o ID do capacete (que na verdade é o nosso boné com tela)
EPI_CAPACETE_ID = 2

# YOLO World consegue identificar "cap" e "hat", o que é perfeito pro boné!
CLASSES_YOLO = [
    "hard hat", "helmet", "safety helmet", "cap", "baseball cap", "hat",
    "person"
]

HELMET_CLASSES = [0, 1, 2, 3, 4, 5]
PERSON_CLASS = 6
LIMITE_CONFIANCA_FACE = 60

# ==============================================================================
# VARIÁVEIS GLOBAIS
# ==============================================================================
camera_ativa = True
nomes_conhecidos = {}
modelo_treinado = False
tempo_infracao = {}

frame_atual = None
lock_frame = threading.Lock()

ultimo_desenho_capacetes = []

foco_nome = "Desconhecido"
foco_status = "ANALISANDO..."
foco_cor = (255, 255, 0)
foco_bbox = None
tempo_ultimo_treino = time.time()

# Caminhos Locais (Restaurado para Legado)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASENAME_AMOSTRAS = os.path.join(BASE_DIR, "Trabalho-E.P.I", "reconhecimento_facial", "amostras_faciais_arquivos")

# ==============================================================================
# 2. INICIALIZAÇÃO DOS MODELOS (YOLO E FACIAL)
# ==============================================================================
print("[SISTEMA] Carregando Modelos YOLO e HaarCascades...")
model = YOLO("yolov8s-worldv2.pt")
model.set_classes(CLASSES_YOLO)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_profileface.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# ==============================================================================
# 3. FUNÇÕES DE SUPORTE E BANCO DE DADOS (NOVO SCHEMA epi_guard)
# ==============================================================================

def treinar_modelo():
    global modelo_treinado, nomes_conhecidos
    try:
        print("[SISTEMA] Conectando ao banco para carregar funcionarios...")
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        cursor.execute("SELECT id, nome FROM funcionarios WHERE status = 'ATIVO'")
        nomes_conhecidos = {row[0]: row[1] for row in cursor.fetchall()}
        print(f"[SISTEMA] {len(nomes_conhecidos)} funcionarios ativos carregados.")

        cursor.execute("SELECT funcionario_id, caminho_imagem FROM amostras_faciais")
        faces, ids = [], []
        for uid, caminho in cursor.fetchall():
            if not caminho: continue
            
            p_final = None
            opcoes = [
                caminho,
                os.path.join(BASENAME_AMOSTRAS, os.path.basename(caminho)),
                os.path.join(os.getcwd(), caminho)
            ]
            for opt in opcoes:
                if os.path.exists(opt):
                    p_final = opt
                    break
                
            if p_final:
                img = cv2.imread(p_final, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    faces.append(cv2.resize(img, (200, 200)))
                    ids.append(uid)

        if len(faces) > 0:
            recognizer.train(faces, np.array(ids))
            modelo_treinado = True
            print(f"[TREINO] Modelo facial treinado com {len(faces)} amostras.")
        else:
            modelo_treinado = False
            print("[ALERTA] Nenhuma amostra facial encontrada. O reconhecimento nao funcionara.")
        conn.close()
    except Exception as e:
        print(f"[ERRO BANCO] Falha ao conectar ou treinar: {e}")


def registrar_multa(frame_evidencia, funcionario_id, falta_capacete):
    """
    Novo schema epi_guard focado Apenas no Capacete Azul
    """
    if not falta_capacete:
        return # Só registra se faltar o boné
        
    try:
        nome_funcionario = nomes_conhecidos.get(funcionario_id, f"ID {funcionario_id}")
        print(f"[IA] Detectada infracao para: {nome_funcionario}")

        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # 1. Salvar imagem como arquivo
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        nome_arquivo = f"evidencia_{funcionario_id}_{timestamp}.jpg"
        caminho_completo = os.path.join(PASTA_EVIDENCIAS, nome_arquivo)
        cv2.imwrite(caminho_completo, frame_evidencia)

        # 2. Criar uma única ocorrência do tipo INFRACAO
        cursor.execute(
            "INSERT INTO ocorrencias (funcionario_id, data_hora, tipo) VALUES (%s, NOW(), 'INFRACAO')",
            (funcionario_id,)
        )
        ocorrencia_id = cursor.lastrowid

        # 3. Inserir a ausência do EPI (Boné/Capacete)
        cursor.execute(
            "INSERT INTO ocorrencia_epis (ocorrencia_id, epi_id) VALUES (%s, %s)",
            (ocorrencia_id, EPI_CAPACETE_ID)
        )

        # 4. Inserir evidência com caminho do arquivo
        cursor.execute(
            "INSERT INTO evidencias (ocorrencia_id, caminho_imagem) VALUES (%s, %s)",
            (ocorrencia_id, caminho_completo)
        )

        conn.commit()
        conn.close()

        print(f"[SUCESSO] Ocorrencia #{ocorrencia_id} registrada no banco. Faltando: CAPACETE (Boné)")
        
        threading.Thread(target=lambda: winsound.Beep(2500, 1000)).start()
    except Exception as e:
        print(f"[ERRO REGISTRO] {e}")


def verificar_hsv_capacete(img_crop):
    if img_crop is None or img_crop.size == 0:
        return False, 0.0, 0.0
    h, w = img_crop.shape[:2]
    
    hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
    
    # 1. Característica 1: O detalhe central e aba AZUL ESCURO
    # Aumentei a margem do matiz de azul para garantir que a webcam não perca o tom
    lower_blue = np.array([80, 40, 20])
    upper_blue = np.array([150, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # 2. Característica 2: Laterais em tela PRETA
    # Preto tem V (Value/Brilho) muito baixo. Ampliei a tolerância de brilho pra telas cinza/iluminadas
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 80])
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    
    kernel = np.ones((3, 3), np.uint8)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
    mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_OPEN, kernel)
    
    area_total = h * w
    ratio_blue = cv2.countNonZero(mask_blue) / area_total
    ratio_black = cv2.countNonZero(mask_black) / area_total
    
    # Tolerâncias mais flexíveis para evitar falhas com luz ambiente (3% de azul, 8% de preto)
    tem_azul = ratio_blue >= 0.02   
    tem_preto = ratio_black >= 0.05 
    
    return (tem_azul and tem_preto), ratio_blue, ratio_black

# ==============================================================================
# 4. THREADS DE VÍDEO E IA
# ==============================================================================

def capturar_frames():
    global frame_atual, camera_ativa
    cap = None
    while True:
        if camera_ativa:
            if cap is None:
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                cap.set(3, 1280)
                cap.set(4, 720)
                if not cap.isOpened():
                    print("[ERRO] Nao foi possivel abrir a camera.")
                    cap = None
                    camera_ativa = False
                    continue
                print("[SISTEMA] Webcam iniciada com sucesso.")

            ret, frame = cap.read()
            if ret:
                with lock_frame:
                    frame_atual = frame.copy()
            else:
                time.sleep(0.01)
        else:
            if cap is not None:
                cap.release()
                cap = None
                with lock_frame:
                    frame_atual = None
                print("[SISTEMA] Webcam encerrada.")
            time.sleep(0.5)


def processar_ia():
    global frame_atual, camera_ativa
    global ultimo_desenho_capacetes
    global foco_nome, foco_status, foco_cor, foco_bbox
    global tempo_infracao, modelo_treinado, nomes_conhecidos, tempo_ultimo_treino

    frame_count = 0
    print("[SISTEMA] IA de deteccao de EPI (Somente Bone) em execucao...")

    while True:
        if not camera_ativa:
            tempo_infracao.clear()
            foco_bbox = None
            time.sleep(0.5)
            continue

        if frame_atual is None:
            time.sleep(0.01)
            continue

        with lock_frame:
            frame = frame_atual.copy()

        frame_count += 1
        if frame_count % 3 != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Otimizar processamento só buscando as 4 classes
        results = model.predict(frame, conf=0.20, verbose=False, imgsz=800)

        pessoas_yolo, capacetes = [], []

        for r in results:
            for box in r.boxes:
                coords = list(map(int, box.xyxy[0]))
                cls = int(box.cls[0])
                if cls == PERSON_CLASS:
                    pessoas_yolo.append(coords)
                elif cls in HELMET_CLASSES:
                    capacetes.append(coords)

        pessoa_foco = None
        maior_area = 0
        for p in pessoas_yolo:
            area = (p[2]-p[0]) * (p[3]-p[1])
            if area > maior_area:
                maior_area = area
                pessoa_foco = p

        temp_capacetes = []
        temp_foco_bbox = None
        temp_foco_nome = "Desconhecido"
        temp_foco_status = "ANALISANDO..."
        temp_foco_cor = (255, 255, 0)

        if pessoa_foco is not None:
            temp_foco_bbox = pessoa_foco
            px1, py1, px2, py2 = pessoa_foco
            h_img, w_img = frame.shape[:2]
            px1, py1 = max(0, px1), max(0, py1)
            px2, py2 = min(w_img, px2), min(h_img, py2)

            roi_gray = gray[py1:py2, px1:px2]
            
            # Detecção de Face
            faces_haar = list(face_cascade.detectMultiScale(roi_gray, 1.05, 5, minSize=(60, 60)))
            perfil_esq = profile_cascade.detectMultiScale(roi_gray, 1.05, 2, minSize=(40, 40))
            for p in perfil_esq: faces_haar.append(p)
            
            roi_flipped = cv2.flip(roi_gray, 1)
            perfil_dir = profile_cascade.detectMultiScale(roi_flipped, 1.05, 2, minSize=(40, 40))
            if len(perfil_dir) > 0:
                rw_roi = roi_gray.shape[1]
                for (x, y, w_p, h_p) in perfil_dir:
                    faces_haar.append((rw_roi - x - w_p, y, w_p, h_p))

            identidade_id = None
            if len(faces_haar) > 0 and modelo_treinado:
                (fx, fy, fw, fh) = max(faces_haar, key=lambda b: b[2]*b[3])
                try:
                    roi_face = cv2.resize(roi_gray[fy:fy+fh, fx:fx+fw], (200, 200))
                    roi_face = clahe.apply(roi_face)
                    uid, dist = recognizer.predict(roi_face)
                    if dist < LIMITE_CONFIANCA_FACE:
                        identidade_id = uid
                        temp_foco_nome = nomes_conhecidos.get(uid, f"ID {uid}")
                except:
                    pass

            # Auto-atualizar o modelo a cada 5 minutos
            agora_treino = time.time()
            if agora_treino - tempo_ultimo_treino > 300:
                threading.Thread(target=treinar_modelo).start()
                tempo_ultimo_treino = agora_treino

            h_person = py2 - py1
            zona_cabeca = py1 + (h_person * 0.40) # Boné cobre até um pouco mais baixo

            tem_capacete = False

            # Validação apenas do EPI principal (Boné)
            for (hx1, hy1, hx2, hy2) in capacetes:
                hcx = (hx1 + hx2) / 2
                if px1 < hcx < px2 and hy1 < zona_cabeca:
                    valido, r_blue, r_black = verificar_hsv_capacete(frame[hy1:hy2, hx1:hx2])
                    msg_debug = f"Azul:{r_blue:.1%} Preto:{r_black:.1%}"
                    if valido:
                        tem_capacete = True
                        temp_capacetes.append((hx1, hy1, hx2, hy2, msg_debug))
                        break # Achou um bone azul válido na cabeça, não precisa olhar outros boxes
                    else:
                        temp_capacetes.append((hx1, hy1, hx2, hy2, "FALHOU: " + msg_debug))


            falha = not tem_capacete
            temp_foco_cor = (0, 255, 0)
            temp_foco_status = "APROVADO [BONE DETECTADO]"

            if falha:
                temp_foco_cor = (0, 0, 255)
                temp_foco_status = "INFRACAO [SEM BONE AZUL]"

                if identidade_id:
                    agora = time.time()
                    if identidade_id not in tempo_infracao:
                        tempo_infracao[identidade_id] = agora
                    elif agora - tempo_infracao[identidade_id] > 3.0:
                        threading.Thread(target=registrar_multa, args=(
                            frame.copy(), identidade_id, True)).start()
                        tempo_infracao[identidade_id] = agora + 12
                elif frame_count % 30 == 0:
                     print(f"[IA] Infracao para PESSOA DESCONHECIDA (Sem Bone Azul).")
            else:
                if identidade_id in tempo_infracao:
                    del tempo_infracao[identidade_id]

        foco_bbox = temp_foco_bbox
        foco_nome = temp_foco_nome
        foco_status = temp_foco_status
        foco_cor = temp_foco_cor

        ultimo_desenho_capacetes = temp_capacetes

        time.sleep(0.01)


# ==============================================================================
# 5. EXIBIÇÃO NO TERMINAL (cv2.imshow)
# ==============================================================================

def exibir_janela():
    """Exibe o frame processado em uma janela OpenCV. Pressione 'q' para sair."""
    global camera_ativa

    while camera_ativa:
        if frame_atual is None:
            time.sleep(0.05)
            continue

        with lock_frame:
            frame_display = frame_atual.copy()

        # Desenho apenas do Capacete/Boné com Debug de Cores
        for item in ultimo_desenho_capacetes:
            if len(item) == 5:
                hx1, hy1, hx2, hy2, msg = item
            else:
                hx1, hy1, hx2, hy2 = item
                msg = "BONE OK"
                
            cor_box = (255, 100, 0) if "FALHOU" not in msg else (0, 0, 255)
            cv2.rectangle(frame_display, (hx1, hy1), (hx2, hy2), cor_box, 2) 
            cv2.putText(frame_display, msg, (hx1, hy1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cor_box, 1)


        if foco_bbox is not None:
            px1, py1, px2, py2 = foco_bbox
            cv2.rectangle(frame_display, (px1, py1), (px2, py2), foco_cor, 2)
            cv2.putText(frame_display, f"{foco_nome} | {foco_status}",
                        (px1, py1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, foco_cor, 2)

        cv2.imshow("EPI Guard - MONITORAMENTO (Apenas Bone Azul)", frame_display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[SISTEMA] Encerrando...")
            camera_ativa = False
            break

        # Impede fechamento forçado com 'X' fechando a janela de quebrar tudo
        if cv2.getWindowProperty("EPI Guard - MONITORAMENTO (Apenas Bone Azul)", cv2.WND_PROP_VISIBLE) < 1:
            camera_ativa = False
            break

    cv2.destroyAllWindows()


# ==============================================================================
# 6. MAIN — EXECUÇÃO
# ==============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 55)
    print("   EPI GUARD v2.5 - FOCO EM BONE ESCURO")
    print("   Database: epi_guard (Port: 3306)")
    print("   Janela OpenCV: Pressione 'Q' para Sair")
    print("=" * 55 + "\n")

    treinar_modelo()

    threading.Thread(target=capturar_frames, daemon=True).start()
    threading.Thread(target=processar_ia, daemon=True).start()

    exibir_janela()
    print("[SISTEMA] Monitoramento encerrado.")
