import cv2
import mysql.connector
import numpy as np
import threading
import time
import os

# ==============================================================================
# 1. CONFIGURAÇÕES DO BANCO DE DADOS (NOVO SCHEMA epi_guard)
# ==============================================================================
DB_CONFIG = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '',
    'database': 'epi_guard',
    'port': 3308
}

# Garantir que o script rode no seu próprio diretório
os.chdir(os.path.dirname(os.path.abspath(__file__)))

PASTA_AMOSTRAS = os.path.join(os.getcwd(), "amostras_faciais_arquivos")
os.makedirs(PASTA_AMOSTRAS, exist_ok=True)

# ==============================================================================
# VARIÁVEIS GLOBAIS
# ==============================================================================
camera_ativa = True
frame_atual = None
lock_frame = threading.Lock()

foco_nome = "Buscando rosto..."
foco_cor = (0, 0, 255)
modelo_treinado = False
nomes_conhecidos = {}

MODO_CADASTRO = False
FASE_CADASTRO = 1 # 1: Frente, 2: Esq, 3: Dir
MODO_ENTRADA_ID = False
MODO_ENTRADA_NOME = False
ID_DIGITADO = ""
NOME_DIGITADO = ""
cadastro_count = 0
cad_id = 0
cad_nome = ""
ultimo_clique = 0 

LIMITE_CONFIANCA_FACE = 60
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# ==============================================================================
# 2. INICIALIZAÇÃO DOS MODELOS (LBPH e Cascades)
# ==============================================================================
print("[SISTEMA] Carregando Modelos de Face...", flush=True)
cascade_frente = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cascade_perfil = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()


def treinar_modelo():
    global modelo_treinado, nomes_conhecidos
    try:
        print("[SISTEMA] Conectando ao banco para carregar dados...", flush=True)
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, nome FROM funcionarios WHERE status = 'ATIVO'")
        nomes_conhecidos = {row[0]: row[1] for row in cursor.fetchall()}

        cursor.execute("SELECT funcionario_id, caminho_imagem FROM amostras_faciais")
        faces, ids = [], []
        for uid, caminho in cursor.fetchall():
            if not caminho: continue
            
            # Fallback de Caminhos (Resiliência para diferentes locais de execução)
            p_final = None
            opcoes = [
                caminho,
                os.path.join("..", caminho),
                caminho.replace("site novo(facial)/", "").replace("site novo(facial)\\", ""),
                os.path.join(PASTA_AMOSTRAS, os.path.basename(caminho))
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
            # SALVAR O MODELO PARA O SITE USAR
            path_yml = os.path.join(os.path.dirname(os.path.abspath(__file__)), "treinador.yml")
            recognizer.write(path_yml)
            
            modelo_treinado = True
            print(f"[TREINO] Reconhecimento pronto ({len(faces)} amostras) - Gravado treinador.yml", flush=True)
        else:
            modelo_treinado = False
            print("[TREINO] Nenhuma amostra encontrada.", flush=True)
        conn.close()
    except Exception as e:
        print(f"[ERRO TREINO] {e}", flush=True)

# ==============================================================================
# 3. THREADS DE CAPTURA
# ==============================================================================

def capturar_frames():
    global frame_atual, camera_ativa
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("[ERRO] Nao foi possivel abrir a camera.")
        camera_ativa = False
        return

    while camera_ativa:
        ret, frame = cap.read()
        if ret:
            with lock_frame:
                frame_atual = frame.copy()
        time.sleep(0.01)
    cap.release()


def salvar_amostra(face_img, uid):
    try:
        timestamp = int(time.time() * 1000)
        nome_arquivo = f"func_{uid}_{timestamp}.jpg"
        caminho_total = os.path.join(PASTA_AMOSTRAS, nome_arquivo)
        cv2.imwrite(caminho_total, face_img)
        
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Verificar se o funcionario existe antes de inserir (evitar erro de FK)
        cursor.execute("SELECT id FROM funcionarios WHERE id = %s", (uid,))
        if not cursor.fetchone():
            print(f"[ERRO] Funcionario ID {uid} nao existe no banco!")
            conn.close()
            return

        caminho_db = os.path.join("amostras_faciais_arquivos", nome_arquivo)
        cursor.execute(
            "INSERT INTO amostras_faciais (funcionario_id, caminho_imagem, criado_em) VALUES (%s, %s, NOW())",
            (uid, caminho_db)
        )
        conn.commit()
        conn.close()
        # print(f"[OK] Amostra salva para ID {uid}")
    except Exception as e:
        print(f"[ERRO SALVAR] {e}", flush=True)


# ==============================================================================
# 4. LOOP PRINCIPAL - TERMINAL
# ==============================================================================

def iniciar_v2():
    global frame_atual, foco_nome, foco_cor, camera_ativa
    global MODO_CADASTRO, FASE_CADASTRO, MODO_ENTRADA_ID, MODO_ENTRADA_NOME
    global ID_DIGITADO, NOME_DIGITADO, cadastro_count, cad_id, cad_nome, ultimo_clique
    
    print("\n" + "=" * 55)
    print("   EPI GUARD - RECONHECIMENTO FACIAL TERMINAL")
    print("   Controle pela TELA da camera!")
    print("=" * 55 + "\n")

    while camera_ativa:
        if frame_atual is None:
            time.sleep(0.01)
            continue

        with lock_frame:
            frame = frame_atual.copy()
        
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detecção de Face (Frente e Perfil)
        # SENSITIVIDADE MÁXIMA (1.05) para capturar angulos intermediarios
        caixas_frente = list(cascade_frente.detectMultiScale(gray, 1.05, 5, minSize=(80, 80)))
        # Detector frontal permissivo para angulos de transição
        caixas_frente_flex = list(cascade_frente.detectMultiScale(gray, 1.1, 3, minSize=(70, 70)))
        
        caixas_esq = list(cascade_perfil.detectMultiScale(gray, 1.05, 2, minSize=(60, 60)))
        
        gray_flipped = cv2.flip(gray, 1)
        perfil_dir_raw = cascade_perfil.detectMultiScale(gray_flipped, 1.05, 2, minSize=(60, 60))
        caixas_dir = []
        for (x, y, w_p, h_p) in perfil_dir_raw:
            caixas_dir.append((w - x - w_p, y, w_p, h_p))

        # Combinar para visualização geral
        caixas_todas = caixas_frente + caixas_esq + caixas_dir
        
        rosto_foco = None
        if MODO_CADASTRO:
            # No modo cadastro, usamos lógica híbrida para capturar a transição
            if FASE_CADASTRO == 1:
                if caixas_frente:
                    rosto_foco = max(caixas_frente, key=lambda b: b[2]*b[3])
                fase_msg = "OLHE PARA FRENTE"
            
            elif FASE_CADASTRO == 2:
                # Prioriza perfil esquerdo, mas aceita frontal flexível (transição)
                if caixas_esq:
                    rosto_foco = max(caixas_esq, key=lambda b: b[2]*b[3])
                elif caixas_frente_flex:
                    rosto_foco = max(caixas_frente_flex, key=lambda b: b[2]*b[3])
                fase_msg = "VIRE LENTAMENTE PARA ESQUERDA..."
            
            elif FASE_CADASTRO == 3:
                # Prioriza perfil direito, mas aceita frontal flexível (transição)
                if caixas_dir:
                    rosto_foco = max(caixas_dir, key=lambda b: b[2]*b[3])
                elif caixas_frente_flex:
                    rosto_foco = max(caixas_frente_flex, key=lambda b: b[2]*b[3])
                fase_msg = "VIRE LENTAMENTE PARA DIREITA..."
            
            if rosto_foco is None:
                fase_msg = "POSICIONE O ROSTO..." if FASE_CADASTRO==1 else "VIRE O ROSTO AOS POUCOS..."
            
            if fase_msg:
                cv2.putText(frame, fase_msg, (w//2-150, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        
        elif caixas_todas:
            rosto_foco = max(caixas_todas, key=lambda b: b[2]*b[3])

        # ----------------------------------------------------------------------
        # Interface de Entrada de Dados na Tela (ID ou NOME)
        # ----------------------------------------------------------------------
        if MODO_ENTRADA_ID or MODO_ENTRADA_NOME:
            # Sombra/Fundo para o input
            cv2.rectangle(frame, (w//2-250, h//2-60), (w//2+250, h//2+60), (0,0,0), -1)
            cv2.rectangle(frame, (w//2-250, h//2-60), (w//2+250, h//2+60), (255,255,255), 2)
            
            if MODO_ENTRADA_ID:
                msg = "DIGITE O ID E ENTER:"
                val = ID_DIGITADO
            else:
                msg = f"ID {cad_id} - DIGITE O NOME:"
                val = NOME_DIGITADO

            cv2.putText(frame, msg, (w//2-230, h//2-15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, f"> {val}_", (w//2-230, h//2+30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        
        elif rosto_foco is not None:
            rx, ry, rw, rh = rosto_foco
            roi_gray = gray[ry:ry+int(rh*0.7), rx:rx+rw]
            
            if roi_gray.size > 0:
                roi_proc = cv2.resize(roi_gray, (200, 200))
                # Normalização de Luz (CLAHE)
                roi_proc = clahe.apply(roi_proc)

                if MODO_CADASTRO:
                    # Captura mais balanceada (1 a cada 0.2s)
                    agora = time.time()
                    if agora - ultimo_clique > 0.2:
                        if cadastro_count < 20: # 20 por fase (Total 60)
                            cadastro_count += 1
                            salvar_amostra(roi_proc, cad_id)
                            foco_nome = f"FASE {FASE_CADASTRO}/3 - CAPTURA {cadastro_count}/20"
                            foco_cor = (0, 165, 255)
                            ultimo_clique = agora
                        elif FASE_CADASTRO < 3:
                            FASE_CADASTRO += 1
                            cadastro_count = 0
                            print(f"[SISTEMA] Indo para fase {FASE_CADASTRO}...")
                        else:
                            print(f"[SUCESSO] ID {cad_id} cadastrado com 60 amostras.")
                            MODO_CADASTRO = False
                            treinar_modelo()
                
                elif modelo_treinado:
                    uid, conf = recognizer.predict(roi_proc)
                    if conf < LIMITE_CONFIANCA_FACE:
                        foco_nome = nomes_conhecidos.get(uid, f"ID {uid}")
                        foco_cor = (0, 255, 0)
                    else:
                        foco_nome = "Desconhecido"
                        foco_cor = (0, 0, 255)
            
            cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), foco_cor, 2)
            cv2.putText(frame, foco_nome, (rx, ry-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, foco_cor, 2)

        if not MODO_ENTRADA_ID and not MODO_CADASTRO:
            cv2.putText(frame, "Pressione 'C' para novo cadastro | 'Q' para Sair", (20, h-30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        cv2.imshow("RECONHECIMENTO FACIAL - EPI GUARD", frame)
        key = cv2.waitKey(1) & 0xFF

        if key != 255: # Ignorar quando nenhuma tecla e pressionada (255 e o padrao de cv2.waitKey & 0xFF)
            # LOG DE DEBUG NO TERMINAL
            try:
                char = chr(key) if 32 <= key <= 126 else "N/A"
                print(f"[DEBUG] Tecla: {key} | Caractere: {char}", flush=True)
            except:
                pass

        if key == ord('q'):
            camera_ativa = False
        
        elif MODO_ENTRADA_ID:
            if key == 13 or key == 10: # ENTER
                if ID_DIGITADO:
                    try:
                        cad_id = int(ID_DIGITADO)
                        conn = mysql.connector.connect(**DB_CONFIG)
                        cursor = conn.cursor()
                        cursor.execute("SELECT nome FROM funcionarios WHERE id = %s", (cad_id,))
                        res = cursor.fetchone()
                        conn.close()
                        
                        if res:
                            cad_nome = res[0]
                            print(f"[SISTEMA] ID {cad_id} ({cad_nome}) encontrado. Adicionando fotos...")
                            MODO_ENTRADA_ID = False
                            MODO_CADASTRO = True
                            FASE_CADASTRO = 1
                            cadastro_count = 0
                        else:
                            print(f"[SISTEMA] ID {cad_id} nao existe. Criando novo...")
                            MODO_ENTRADA_ID = False
                            MODO_ENTRADA_NOME = True
                            NOME_DIGITADO = ""
                    except:
                        ID_DIGITADO = ""
            elif key == 8: # BACKSPACE
                ID_DIGITADO = ID_DIGITADO[:-1]
            elif 48 <= key <= 57: # 0-9
                ID_DIGITADO += chr(key)
            elif key == 27: # ESC
                MODO_ENTRADA_ID = False

        elif MODO_ENTRADA_NOME:
            if key == 13 or key == 10: # ENTER
                if NOME_DIGITADO:
                    cad_nome = NOME_DIGITADO
                    try:
                        # INSERIR NOVO FUNCIONARIO
                        conn = mysql.connector.connect(**DB_CONFIG)
                        cursor = conn.cursor()
                        cursor.execute(
                            "INSERT INTO funcionarios (id, nome, status, criado_em) VALUES (%s, %s, 'ATIVO', NOW())",
                            (cad_id, cad_nome)
                        )
                        conn.commit()
                        conn.close()
                        print(f"[SUCESSO] Funcionario {cad_nome} criado no banco.")
                        MODO_ENTRADA_NOME = False
                        MODO_CADASTRO = True
                        FASE_CADASTRO = 1
                        cadastro_count = 0
                    except Exception as e:
                        print(f"[ERRO] Falha ao criar funcionario: {e}")
                        MODO_ENTRADA_NOME = False
            elif key == 8: # BACKSPACE
                NOME_DIGITADO = NOME_DIGITADO[:-1]
            elif 32 <= key <= 126: # Letras, numeros e espaco
                NOME_DIGITADO += chr(key).upper()
            elif key == 27: # ESC
                MODO_ENTRADA_NOME = False

        elif key == ord('c') and not MODO_CADASTRO:
            MODO_ENTRADA_ID = True
            ID_DIGITADO = ""

    cv2.destroyAllWindows()


if __name__ == '__main__':
    import sys
    if "--train-only" in sys.argv:
        treinar_modelo()
    else:
        treinar_modelo()
        threading.Thread(target=capturar_frames, daemon=True).start()
        iniciar_v2()
