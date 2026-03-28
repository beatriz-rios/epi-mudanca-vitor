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
    'database': 'epi_guard',
    'port': 3308
}

PASTA_EVIDENCIAS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evidencias")
os.makedirs(PASTA_EVIDENCIAS, exist_ok=True)

EPI_OCULOS_ID = 1
EPI_CAPACETE_ID = 2
EPI_CORPO_ID = 3
EPI_LUVAS_ID = 4

CLASSES_YOLO = [
    "hard hat", "helmet", "safety helmet",
    "person",
    "glasses", "sunglasses", "reading glasses",
    "safety goggles", "protective eyewear", "safety glasses",
    "welding jacket", "leather jacket", "protective jacket",
    "welding apron", "leather apron", "apron",
    "glove", "gloves"
]

BLUSAO_CLASSES = [10, 11, 12]
AVENTAL_CLASSES = [13, 14, 15]
GLOVE_CLASSES = [16, 17]
HELMET_CLASSES = [0, 1, 2]
PERSON_CLASS = 3
ALL_EYEWEAR = [4, 5, 6, 7, 8, 9]
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
ultimo_desenho_oculos = []
ultimo_desenho_oculos_vermelho = []
ultimo_desenho_blusoes = []
ultimo_desenho_aventais = []
ultimo_desenho_luvas = []

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
            
            # Fallback de Caminhos (Legado)
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


def registrar_multa(frame_evidencia, funcionario_id, falta_capacete, falta_oculos, falta_corpo, falta_luvas):
    """
    Novo schema epi_guard:
    1. Salva imagem como arquivo JPG na pasta evidencias/
    2. Cria UMA ocorrência com tipo='INFRACAO'
    3. Insere cada EPI faltante em ocorrencia_epis
    4. Insere evidência com caminho do arquivo
    """
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

        # 3. Inserir cada EPI faltante na tabela ocorrencia_epis
        epis_faltantes = []
        if falta_capacete: epis_faltantes.append(EPI_CAPACETE_ID)
        if falta_oculos: epis_faltantes.append(EPI_OCULOS_ID)
        if falta_corpo: epis_faltantes.append(EPI_CORPO_ID)
        if falta_luvas: epis_faltantes.append(EPI_LUVAS_ID)

        for epi_id in epis_faltantes:
            cursor.execute(
                "INSERT INTO ocorrencia_epis (ocorrencia_id, epi_id) VALUES (%s, %s)",
                (ocorrencia_id, epi_id)
            )

        # 4. Inserir evidência com caminho do arquivo
        cursor.execute(
            "INSERT INTO evidencias (ocorrencia_id, caminho_imagem) VALUES (%s, %s)",
            (ocorrencia_id, caminho_completo)
        )

        conn.commit()
        conn.close()

        nomes_epis = []
        if falta_capacete: nomes_epis.append("CAPACETE")
        if falta_oculos: nomes_epis.append("OCULOS")
        if falta_corpo: nomes_epis.append("BLUSAO/AVENTAL")
        if falta_luvas: nomes_epis.append("LUVAS")

        print(f"[SUCESSO] Ocorrencia #{ocorrencia_id} registrada no banco.")
        print(f"         Faltando: {', '.join(nomes_epis)}")
        
        threading.Thread(target=lambda: winsound.Beep(2500, 1000)).start()
    except Exception as e:
        print(f"[ERRO REGISTRO] {e}")


def verificar_hsv_capacete(img_crop):
    if img_crop is None or img_crop.size == 0:
        return False
    h, w = img_crop.shape[:2]
    topo = img_crop[0:int(h*0.7), :]
    hsv = cv2.cvtColor(topo, cv2.COLOR_BGR2HSV)
    # Ajuste fino para detectar capacetes de segurança (amarelo, branco, cinza, etc)
    mask_valid = cv2.inRange(hsv, np.array([0, 0, 50]), np.array([180, 255, 255]))
    ratio = cv2.countNonZero(mask_valid) / (topo.shape[0]*topo.shape[1])
    return ratio > 0.35


def verificar_cor_epi_oculos(img_crop):
    if img_crop is None or img_crop.size == 0:
        return False
    img_crop = cv2.resize(img_crop, (220, 100))
    img_crop = cv2.GaussianBlur(img_crop, (3, 3), 0)
    hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)

    # Detecção de óculos com hastes amarelas ou detalhes vermelhos (padrão EPI)
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([38, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    lower_red1 = np.array([0, 130, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 130, 70])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

    kernel = np.ones((3, 3), np.uint8)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)

    total_amarelo = cv2.countNonZero(mask_yellow)
    total_vermelho = cv2.countNonZero(mask_red)

    area_total = img_crop.shape[0] * img_crop.shape[1]
    percentual_amarelo = (total_amarelo / area_total) * 100
    percentual_vermelho = (total_vermelho / area_total) * 100

    if percentual_amarelo > 0.8 or percentual_vermelho > 0.5:
        return True
    return False

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
    global ultimo_desenho_capacetes, ultimo_desenho_oculos, ultimo_desenho_oculos_vermelho
    global ultimo_desenho_blusoes, ultimo_desenho_aventais, ultimo_desenho_luvas
    global foco_nome, foco_status, foco_cor, foco_bbox
    global tempo_infracao, modelo_treinado, nomes_conhecidos, tempo_ultimo_treino

    frame_count = 0
    print("[SISTEMA] IA de deteccao de EPI em execucao...")

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
        results = model.predict(frame, conf=0.20, verbose=False, imgsz=800)

        pessoas_yolo, capacetes, oculos_detectados = [], [], []
        blusoes_detectados, aventais_detectados = [], []
        luvas_detectadas_raw = []

        for r in results:
            for box in r.boxes:
                coords = list(map(int, box.xyxy[0]))
                cls = int(box.cls[0])
                if cls == PERSON_CLASS:
                    pessoas_yolo.append(coords)
                elif cls in HELMET_CLASSES:
                    capacetes.append(coords)
                elif cls in ALL_EYEWEAR:
                    oculos_detectados.append(coords)
                elif cls in BLUSAO_CLASSES:
                    blusoes_detectados.append(coords)
                elif cls in AVENTAL_CLASSES:
                    aventais_detectados.append(coords)
                elif cls in GLOVE_CLASSES:
                    luvas_detectadas_raw.append(coords)

        pessoa_foco = None
        maior_area = 0
        for p in pessoas_yolo:
            area = (p[2]-p[0]) * (p[3]-p[1])
            if area > maior_area:
                maior_area = area
                pessoa_foco = p

        temp_capacetes = []
        temp_oculos = []
        temp_oculos_vermelho = []
        temp_blusoes = []
        temp_aventais = []
        luvas_filtradas = []

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
            
            # Detecção de Face (Frente + Perfil Esquerdo + Perfil Direito)
            # SENSITIVIDADE MÁXIMA (1.05) para capturar angulos intermediarios
            faces_haar = list(face_cascade.detectMultiScale(roi_gray, 1.05, 5, minSize=(60, 60)))
            
            # Perfil Esquerdo
            perfil_esq = profile_cascade.detectMultiScale(roi_gray, 1.05, 2, minSize=(40, 40))
            for p in perfil_esq: faces_haar.append(p)
            
            # Perfil Direito (espelhado)
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
                    # Usar o rosto inteiro (100%) para bater com o cadastro
                    roi_face = cv2.resize(roi_gray[fy:fy+fh, fx:fx+fw], (200, 200))
                    # Normalização de Luz (CLAHE) para bater com o cadastro
                    roi_face = clahe.apply(roi_face)
                    uid, dist = recognizer.predict(roi_face)
                    if dist < LIMITE_CONFIANCA_FACE:
                        identidade_id = uid
                        temp_foco_nome = nomes_conhecidos.get(uid, f"ID {uid}")
                except:
                    pass

            # Auto-atualizar o modelo a cada 5 minutos (300 segundos)
            agora_treino = time.time()
            if agora_treino - tempo_ultimo_treino > 300:
                threading.Thread(target=treinar_modelo).start()
                tempo_ultimo_treino = agora_treino

            h_person = py2 - py1
            zona_cabeca = py1 + (h_person * 0.35)
            zona_olhos = py1 + (h_person * 0.55)

            tem_capacete, tem_oculos = False, False
            tem_blusao, tem_avental = False, False

            # Validacao de proximidade dos EPIs a pessoa
            for (hx1, hy1, hx2, hy2) in capacetes:
                hcx = (hx1 + hx2) / 2
                if px1 < hcx < px2 and hy1 < zona_cabeca:
                    if verificar_hsv_capacete(frame[hy1:hy2, hx1:hx2]):
                        tem_capacete = True
                        temp_capacetes.append((hx1, hy1, hx2, hy2))

            for (ox1, oy1, ox2, oy2) in oculos_detectados:
                ocx, ocy = (ox1 + ox2) / 2, (oy1 + oy2) / 2
                if px1 < ocx < px2 and py1 < ocy < zona_olhos:
                    largura = ox2 - ox1
                    margem = int(largura * 0.5)
                    crop_x1 = max(0, ox1 - margem)
                    crop_x2 = min(w_img, ox2 + margem)
                    crop_oculos = frame[oy1:oy2, crop_x1:crop_x2]

                    if verificar_cor_epi_oculos(crop_oculos):
                        tem_oculos = True
                        temp_oculos.append((ox1, oy1, ox2, oy2))
                    else:
                        temp_oculos_vermelho.append((ox1, oy1, ox2, oy2))

            for (bx1, by1, bx2, by2) in blusoes_detectados:
                centro_x = (bx1 + bx2) // 2
                centro_y = (by1 + by2) // 2
                if px1 < centro_x < px2 and py1 < centro_y < py2:
                    tem_blusao = True
                    temp_blusoes.append((bx1, by1, bx2, by2))

            for (ax1, ay1, ax2, ay2) in aventais_detectados:
                centro_x = (ax1 + ax2) // 2
                centro_y = (ay1 + ay2) // 2
                if px1 < centro_x < px2 and py1 < centro_y < py2:
                    tem_avental = True
                    temp_aventais.append((ax1, ay1, ax2, ay2))

            for coords in luvas_detectadas_raw:
                lx1, ly1, lx2, ly2 = coords
                lcx, lcy = (lx1 + lx2) / 2, (ly1 + ly2) / 2
                margem_h = (px2 - px1) * 0.20
                if (px1 - margem_h) < lcx < (px2 + margem_h) and (py1 - 50) < lcy < py2:
                    falso_positivo = False
                    dist_minima = (px2 - px1) * 0.15
                    for (fx1, fy1, fx2, fy2) in luvas_filtradas:
                        fcx, fcy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
                        dist = ((lcx - fcx)**2 + (lcy - fcy)**2)**0.5
                        if dist < dist_minima:
                            falso_positivo = True
                            break
                    if not falso_positivo: luvas_filtradas.append((lx1, ly1, lx2, ly2))

            tem_luvas = (len(luvas_filtradas) >= 2)
            tem_corpo = tem_blusao or tem_avental

            falha = not (tem_capacete and tem_oculos and tem_corpo and tem_luvas)
            temp_foco_cor = (0, 255, 0)
            temp_foco_status = "APROVADO"

            if falha:
                temp_foco_cor = (0, 0, 255)
                temp_foco_status = "INFRACAO"
                if not tem_capacete: temp_foco_status += " [CAPACETE]"
                if not tem_oculos: temp_foco_status += " [OCULOS]"
                if not tem_corpo: temp_foco_status += " [CORPO]"
                if not tem_luvas: temp_foco_status += " [LUVAS]"

                if identidade_id:
                    agora = time.time()
                    if identidade_id not in tempo_infracao:
                        tempo_infracao[identidade_id] = agora
                    elif agora - tempo_infracao[identidade_id] > 3.0:
                        threading.Thread(target=registrar_multa, args=(
                            frame.copy(), identidade_id, not tem_capacete, not tem_oculos, not tem_corpo, not tem_luvas)).start()
                        tempo_infracao[identidade_id] = agora + 12 # Espera 12s para proximo registro
                elif frame_count % 30 == 0:
                     print(f"[IA] Detectada falha para PESSOA DESCONHECIDA. Nao inserida no banco.")

            else:
                if identidade_id in tempo_infracao:
                    del tempo_infracao[identidade_id]

        foco_bbox = temp_foco_bbox
        foco_nome = temp_foco_nome
        foco_status = temp_foco_status
        foco_cor = temp_foco_cor

        ultimo_desenho_capacetes = temp_capacetes
        ultimo_desenho_oculos = temp_oculos
        ultimo_desenho_oculos_vermelho = temp_oculos_vermelho
        ultimo_desenho_blusoes = temp_blusoes
        ultimo_desenho_aventais = temp_aventais
        ultimo_desenho_luvas = luvas_filtradas

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

        # Desenho dos EPIs detectados
        for (hx1, hy1, hx2, hy2) in ultimo_desenho_capacetes:
            cv2.rectangle(frame_display, (hx1, hy1), (hx2, hy2), (0, 255, 0), 2)
        for (ox1, oy1, ox2, oy2) in ultimo_desenho_oculos:
            cv2.rectangle(frame_display, (ox1, oy1), (ox2, oy2), (0, 255, 0), 2)
        for (ox1, oy1, ox2, oy2) in ultimo_desenho_oculos_vermelho:
            cv2.rectangle(frame_display, (ox1, oy1), (ox2, oy2), (0, 0, 255), 2)
        for (bx1, by1, bx2, by2) in ultimo_desenho_blusoes:
            cv2.rectangle(frame_display, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
        for (ax1, ay1, ax2, ay2) in ultimo_desenho_aventais:
            cv2.rectangle(frame_display, (ax1, ay1), (ax2, ay2), (0, 255, 0), 2)
        for (lx1, ly1, lx2, ly2) in ultimo_desenho_luvas:
            cv2.rectangle(frame_display, (lx1, ly1), (lx2, ly2), (0, 255, 0), 2)

        if foco_bbox is not None:
            px1, py1, px2, py2 = foco_bbox
            cv2.rectangle(frame_display, (px1, py1), (px2, py2), foco_cor, 2)
            cv2.putText(frame_display, f"{foco_nome} | {foco_status}",
                        (px1, py1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, foco_cor, 2)

        cv2.imshow("EPI Guard - MONITORAMENTO", frame_display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[SISTEMA] Encerrando...")
            camera_ativa = False
            break

    cv2.destroyAllWindows()


# ==============================================================================
# 6. MAIN — EXECUÇÃO
# ==============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 55)
    print("   EPI GUARD v2.0 - MONITORAMENTO TERMINAL")
    print("   Database: epi_guard (Port: 3308)")
    print("   Janela OpenCV: Pressione 'Q' para Sair")
    print("=" * 55 + "\n")

    treinar_modelo()

    threading.Thread(target=capturar_frames, daemon=True).start()
    threading.Thread(target=processar_ia, daemon=True).start()

    exibir_janela()
    print("[SISTEMA] Monitoramento encerrado.")
