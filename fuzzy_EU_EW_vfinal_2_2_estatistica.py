import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler 
import sys 
import math
from sklearn.metrics import adjusted_rand_score, accuracy_score, fbeta_score, normalized_mutual_info_score

# Aumenta a precisão de impressão do NumPy
np.set_printoptions(precision=6, suppress=True)

# -------------------- FUNÇÃO PARA MATRIZ DE CONFUSÃO --------------------

def generate_confusion_matrix_clustering(y_true, cluster_labels, n_clusters):
    unique_true_classes = np.unique(y_true)
    class_to_index = {cls: i for i, cls in enumerate(unique_true_classes)}
    n_true_classes = len(unique_true_classes)
    
    confusion_matrix = np.zeros((n_clusters, n_true_classes), dtype=int)
    
    for i in range(n_clusters):
        true_labels_in_cluster = y_true[cluster_labels == i]
        for true_class in true_labels_in_cluster:
            j = class_to_index[true_class]
            confusion_matrix[i, j] += 1
            
    col_names = [f'Classe {cls}' for cls in unique_true_classes]
    row_names = [f'Cluster {i}' for i in range(n_clusters)]
    df_cm = pd.DataFrame(confusion_matrix, index=row_names, columns=col_names)
    
    return df_cm

# -------------------- FUNÇÕES DE UTILIDADE E KERNEL --------------------

def load_and_preprocess_data(data_file_path, headers_file_path, delimiter, target_column_name=None, perform_normalization=True):
    try:
        headers_df = pd.read_csv(headers_file_path, header=None)
        headers = headers_df.squeeze().tolist()
        df = pd.read_csv(data_file_path, header=None, names=headers, sep=delimiter)
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado.")
        return None, None, None
    except Exception as e:
        print(f"Erro ao carregar arquivos: {e}")
        return None, None, None

    y_true = None
    if target_column_name and target_column_name in df.columns:
        y_true = df[target_column_name]
        X_df = df.drop(columns=[target_column_name])
    else:
        X_df = df
    
    if perform_normalization:
        scaler = StandardScaler()
        X_processed = scaler.fit_transform(X_df)
        print("Features normalizadas.")
    else:
        X_processed = X_df.values
        print("Dados brutos utilizados.")

    return X_processed, y_true, X_df.columns.tolist()

def local_gaussian_kernel_matrix(X, G, s_squared_matrix):
    """
    Calcula a matriz Kernel K(X, G) onde cada cluster (linha de G) usa seu próprio vetor de larguras s_i.
    X: (N, D)
    G: (K, D)
    s_squared_matrix: (K, D) - Matriz de larguras locais
    
    Retorna:
    K_mat: (N, K) onde K_mat[k, i] é o kernel entre x_k e g_i usando s_i.
    """
    N, D = X.shape
    K_clusters = G.shape[0]
    
    # Prepara o inverso de s^2
    inv_s_squared = np.zeros_like(s_squared_matrix)
    mask = s_squared_matrix > 1e-15
    inv_s_squared[mask] = 1.0 / s_squared_matrix[mask]
    
    # K_mat[k, i] = exp(-0.5 * sum_j (1/s_ij^2) * (x_kj - g_ij)^2)
    K_mat = np.zeros((N, K_clusters))
    
    # Calculamos coluna a coluna (cluster a cluster)
    for i in range(K_clusters):
        # Diferença: (N, D) - (1, D) -> (N, D)
        diff = X - G[i, :]
        diff_sq = diff**2
        
        # Ponderação pela largura do cluster i: (N, D) * (1, D)
        weighted_diff_sq = diff_sq * inv_s_squared[i, :]
        
        # Soma sobre as dimensões
        sum_sq = np.sum(weighted_diff_sq, axis=1) # Resultado (N,)
        
        K_mat[:, i] = np.exp(-0.5 * sum_sq)
        
    return K_mat

# -------------------- FUNÇÃO DE CÁLCULO DE MÉTRICAS --------------------

def compute_evaluation_metrics(y_true, labels, U, n_clusters, beta_val):
    metrics = {}
    y_true_np = y_true.values
    n_samples = len(y_true)
    unique_classes = np.unique(y_true_np)

    # 1. Métricas Externas Básicas
    metrics['ARI'] = adjusted_rand_score(y_true, labels)
    metrics['NMI'] = normalized_mutual_info_score(y_true, labels)

    # 2. Entropia Condicional
    entropy_conditional = 0
    for k in range(n_clusters):
        mask = (labels == k)
        cluster_size = np.sum(mask)
        if cluster_size > 0:
            true_labels_in_cluster = y_true_np[mask]
            cluster_impurity = 0
            for cls in unique_classes:
                p_ij = np.sum(true_labels_in_cluster == cls) / cluster_size
                if p_ij > 0:
                    cluster_impurity -= p_ij * math.log2(p_ij)
            entropy_conditional += (cluster_size / n_samples) * cluster_impurity
    metrics['Entropia_Condicional'] = entropy_conditional

    # 3. Mapeamento para Acurácia e F-measure
    predicted_labels_mapped = np.empty_like(labels, dtype=object)
    for k in range(n_clusters):
        mask = (labels == k)
        if np.sum(mask) > 0:
            true_labels_in_cluster = y_true_np[mask]
            most_frequent_class = pd.Series(true_labels_in_cluster).mode()[0]
            predicted_labels_mapped[mask] = most_frequent_class

    if pd.api.types.is_numeric_dtype(y_true):
        predicted_labels_mapped = predicted_labels_mapped.astype(y_true.dtype)

    metrics['Acuracia'] = accuracy_score(y_true_np, predicted_labels_mapped)
    metrics['ER'] = 1 - metrics['Acuracia']
    
    # F-measure Global e Por Classe
    metrics['F_measure_Weighted'] = fbeta_score(y_true_np, predicted_labels_mapped, beta=beta_val, average='weighted', zero_division=0)
    f_meas_per_class = fbeta_score(y_true_np, predicted_labels_mapped, beta=beta_val, average=None, zero_division=0)
    for cls, score in zip(unique_classes, f_meas_per_class):
        metrics[f'F_measure_Classe_{cls}'] = score

    # =================================================================
    # NOVAS MÉTRICAS DE PARTIÇÃO FUZZY (PC, MPC)
    # =================================================================
    pc = np.sum(U ** 2) / n_samples
    metrics['Partition_Coefficient_PC'] = pc
    
    if n_clusters > 1:
        mpc = 1.0 - (n_clusters / (n_clusters - 1)) * (1.0 - pc)
    else:
        mpc = 0.0
    metrics['Modified_PC_MPC'] = mpc

    # =================================================================
    # MÉTRICAS FUZZY BASEADAS EM PARES (Jaccard, Folkes, Frigui, Hüllermeier)
    # =================================================================
    try:
        # Prepara índices dos pares únicos
        rows, cols = np.triu_indices(n_samples, k=1)
        total_pairs = len(rows)
        
        # Lógica para Jaccard, Folkes-Mallows e Frigui
        Psi_fuzzy = U @ U.T
        y_np = np.asarray(y_true).ravel()
        Psi_true = (y_np[:, None] == y_np[None, :]).astype(float)

        vec_fuzzy = np.ravel(Psi_fuzzy[rows, cols])
        vec_true = np.ravel(Psi_true[rows, cols])

        n_ss_fuzzy = np.sum(vec_fuzzy * vec_true)
        n_sd_fuzzy = np.sum(vec_fuzzy * (1 - vec_true))
        n_ds_fuzzy = np.sum((1 - vec_fuzzy) * vec_true)

        if (n_ss_fuzzy + n_sd_fuzzy + n_ds_fuzzy) > 0:
            metrics['Fuzzy_Jaccard'] = n_ss_fuzzy / (n_ss_fuzzy + n_sd_fuzzy + n_ds_fuzzy)
        else:
            metrics['Fuzzy_Jaccard'] = 0.0

        denom_fm = np.sqrt((n_ss_fuzzy + n_sd_fuzzy) * (n_ss_fuzzy + n_ds_fuzzy))
        if denom_fm > 0:
            metrics['Fuzzy_Folkes_Mallows'] = n_ss_fuzzy / denom_fm
        else:
            metrics['Fuzzy_Folkes_Mallows'] = 0.0

        d_frigui = np.sum((1 - vec_fuzzy) * (1 - vec_true))
        total_pairs_frigui = n_ss_fuzzy + n_sd_fuzzy + n_ds_fuzzy + d_frigui

        if total_pairs_frigui > 0:
            metrics['Fuzzy_Rand_Frigui'] = (n_ss_fuzzy + d_frigui) / total_pairs_frigui
        else:
            metrics['Fuzzy_Rand_Frigui'] = 0.0
            
        # NOVA LÓGICA: Hüllermeier's Rand Variation
        E_Q = (y_np[rows] == y_np[cols]).astype(float)
        
        # Diferença absoluta entre as linhas de pertinência para todos os pares
        diff_U = np.abs(U[rows] - U[cols])
        sum_diff_U = np.sum(diff_U, axis=1)
        
        # Equivalência Fuzzy (E_P)
        E_P = 1.0 - (sum_diff_U / 2.0)
        
        # Somatório das diferenças |E_P - E_Q|
        sum_abs_diff_E = np.sum(np.abs(E_P - E_Q))
        
        if total_pairs > 0:
            rand_hullermeier = 1.0 - (sum_abs_diff_E / total_pairs)
        else:
            rand_hullermeier = 0.0
            
        metrics['Fuzzy_Rand_Hullermeier'] = rand_hullermeier

    except Exception as e:
        print(f"  [Aviso] Erro interno ao calcular métricas fuzzy nesta rodada: {e}")
        metrics['Fuzzy_Jaccard'] = np.nan
        metrics['Fuzzy_Folkes_Mallows'] = np.nan
        metrics['Fuzzy_Rand_Frigui'] = np.nan
        metrics['Fuzzy_Rand_Hullermeier'] = np.nan

    return metrics

# -------------------- CLASSE KFCM-K-W-EwEu.2 --------------------

class KernelFCMWEWEU2:
    """
    Implementa o KFCM-K-W-EwEu.2: Kernel Fuzzy C-Means com Protótipos 
    e cálculo automático de largura LOCAL via Dupla Regularização por Entropia (Tw e Tu).
    """
    def __init__(self, n_clusters=3, Tw=0.1, Tu=0.1, tol=1e-6, max_iter=100, random_state=None):
        self.n_clusters = n_clusters
        self.Tw = Tw 
        self.Tu = Tu
        self.tol = tol
        self.max_iter = max_iter 
        self.random_state = random_state
        self.G = None 
        self.U = None 
        self.s_squared = None # Matriz (K, D)
        
    def get_labels(self):
        if self.U is None:
            return None
        return np.argmax(self.U, axis=1)

    # Função Objetivo J_KFCM-K-W-EwEu.2 (Eq. 32)
    def calculate_objective_function(self, X, G, U, s_squared):
        # Termo 1: Inércia Kernel Local (Sem 'm')
        K_XG = local_gaussian_kernel_matrix(X, G, s_squared) 
        D_XG = 2 * (1 - K_XG)
        term1 = np.sum(U * D_XG)
        
        # Termo 2: Regularização por Entropia das Larguras (Tw)
        inv_s_sq = np.zeros_like(s_squared)
        mask = s_squared > 1e-15
        inv_s_sq[mask] = 1.0 / s_squared[mask]
        
        term_entropy_w = (1 + inv_s_sq) * np.log(1 + inv_s_sq)
        term2 = self.Tw * np.sum(term_entropy_w)
        
        # Termo 3: Regularização por Entropia da Pertinência (Tu)
        term_entropy_u = (1 + U) * np.log(1 + U)
        term3 = self.Tu * np.sum(term_entropy_u)
        
        return term1 + term2 + term3

    # Step 3: Atualiza U (Algoritmo 1 / Eq. 27) - Baseado em Kernels Locais
    def update_U(self, X, G, s_squared):
        N, D = X.shape
        K_clusters = self.n_clusters
        
        K_XG = local_gaussian_kernel_matrix(X, G, s_squared)
        D_XG = 2 * (1 - K_XG)
        
        E_val = np.exp(- D_XG / self.Tu)
        new_U = np.zeros((N, K_clusters))
        
        for k in range(N):
            I_indices = list(range(K_clusters))
            update = True
            
            while update:
                update = False
                card_I = len(I_indices)
                denom_sum = np.sum(E_val[k, I_indices])
                
                if denom_sum < 1e-300:
                    denom_sum = 1e-300
                    
                current_u_vals = {}
                remove_list = []
                
                for i in I_indices:
                    val = ((1 + card_I) * E_val[k, i] / denom_sum) - 1.0
                    current_u_vals[i] = val
                    
                    if val <= 0:
                         remove_list.append(i)
                
                if len(remove_list) > 0:
                    for idx_to_remove in remove_list:
                        if idx_to_remove in I_indices:
                            I_indices.remove(idx_to_remove)
                            new_U[k, idx_to_remove] = 0.0
                    update = True
                else:
                     for i in I_indices:
                        new_U[k, i] = current_u_vals[i]
                        
        return new_U

    # Step 2: Atualiza G (Eq. 21) - Baseado em Kernels Locais
    def update_G(self, X, U, s_squared):
        K_clusters = self.n_clusters
        K_XG = local_gaussian_kernel_matrix(X, self.G, s_squared) 
        new_G = np.zeros_like(self.G)
        
        for i in range(K_clusters):
            # Sem o 'm', usamos apenas U[:, i]
            W_i = U[:, i] * K_XG[:, i] 
            numerator = np.sum(W_i[:, np.newaxis] * X, axis=0)
            denominator = np.sum(W_i)
            
            if denominator < 1e-9:
                new_G[i] = self.G[i].copy()
            else:
                new_G[i] = numerator / denominator

        return new_G

    # Step 1: Atualiza s_squared usando a Fórmula 24 b)
    def update_s_squared(self, X, G, U):
        N, D = X.shape
        K_clusters = self.n_clusters
        
        new_inv_s_squared_matrix = np.zeros((K_clusters, D))
        
        K_XG = local_gaussian_kernel_matrix(X, G, self.s_squared)
        
        # --- Loop sobre cada cluster i ---
        for i in range(K_clusters):
            
            # 1. Calcular Q_ij para este cluster i específico
            diff_sq_i = (X - G[i, :])**2 
            
            # W_ki é o termo (u_ki * K_i) - Sem o 'm'
            W_ki = (U[:, i] * K_XG[:, i])[:, np.newaxis]
            
            # Custo ponderado por feature para o cluster i
            Q_i = np.sum(W_ki * diff_sq_i, axis=0) # (D,)
            
            # 2. Executar iterativo (Active Set) para este cluster i
            J_indices = list(range(D)) 
            update = True
            
            while update:
                update = False
                card_J = len(J_indices)
                
                denom_sum = 0.0
                for h in J_indices:
                    denom_sum += np.exp(- Q_i[h] / self.Tw)
                
                if denom_sum < 1e-300: denom_sum = 1e-300

                current_inv_vals = {}
                remove_list = []
                
                for j in J_indices:
                    # AQUI: Usando apenas 1 + |J| conforme fórmula 24 b) do artigo original
                    num = (1 + card_J) * np.exp(- Q_i[j] / self.Tw)
                    val = (num / denom_sum) - 1.0
                    current_inv_vals[j] = val
                    
                    if val <= 0:
                        remove_list.append(j)
                
                if len(remove_list) > 0:
                    for idx_to_remove in remove_list:
                        if idx_to_remove in J_indices:
                            J_indices.remove(idx_to_remove)
                            # Variável rejeitada recebe diretamente zero
                            new_inv_s_squared_matrix[i, idx_to_remove] = 0.0 
                    update = True
                else:
                    for j in J_indices:
                        new_inv_s_squared_matrix[i, j] = current_inv_vals[j]
        
        # Converte de 1/s^2 para s^2
        new_s_squared = np.zeros((K_clusters, D))
        for i in range(K_clusters):
            for j in range(D):
                if new_inv_s_squared_matrix[i, j] > 1e-15:
                    new_s_squared[i, j] = 1.0 / new_inv_s_squared_matrix[i, j]
                else:
                    new_s_squared[i, j] = 1e15 # Valor "infinito"
                
        return new_s_squared

    def fit(self, X):
        np.random.seed(self.random_state)
        N, D = X.shape
        K_clusters = self.n_clusters
        
        # 1. Inicialização
        random_idx = np.random.choice(N, K_clusters, replace=False)
        self.G = X[random_idx].copy() 
        initial_prototypes = self.G.copy()

        # s_squared Inicialização para KFCM-K-EwEu.2 (LOCAL) com Restrição de Soma:
        # Matriz K x D. sum_j (1/s_ij^2) = 1 para cada i -> Inicializamos com s_ij^2 = D.
        self.s_squared = np.ones((K_clusters, D)) * D
        
        self.U = self.update_U(X, self.G, self.s_squared)
        
        prev_objective_value = self.calculate_objective_function(X, self.G, self.U, self.s_squared)
        print(f"  Iteração 0: J_NEW = {prev_objective_value:.6f}")
        
        iteration_count = 1
        
        while(iteration_count <= self.max_iter):
            J_OLD = prev_objective_value
            
            # Step 1: Atualização de s_squared
            new_s_squared = self.update_s_squared(X, self.G, self.U)
            self.s_squared = new_s_squared
            
            # Step 2: Atualização de Protótipos G (Eq. 21)
            new_G = self.update_G(X, self.U, self.s_squared)
            self.G = new_G
            
            # Step 3: Atualização da Matriz de Pertinência U (Eq. 27)
            new_U = self.update_U(X, self.G, self.s_squared)
            self.U = new_U
            
            current_objective_value = self.calculate_objective_function(X, self.G, self.U, self.s_squared)
            
            print(f"  Iteração {iteration_count}: J_OLD = {J_OLD:.6f}, J_NEW = {current_objective_value:.6f}")
            
            if abs(current_objective_value - J_OLD) < self.tol:
                print(f"  Convergiu! Mudança em J menor que a tolerância ({self.tol}).")
                break
                
            prev_objective_value = current_objective_value
            iteration_count += 1 

        if iteration_count > self.max_iter:
            print(f"\n  Máximo de {self.max_iter} iterações atingido.")

        return initial_prototypes, self.get_labels(), self.U, self.G, self.s_squared, current_objective_value

# -------------------- BLOCO PRINCIPAL --------------------

if __name__ == "__main__":
    
    print("--- Configuração do Dataset (KFCM-K-W-EwEu.2) ---")
    data_file_name = input("Digite o nome do ARQUIVO DE DADOS CSV (ex: processed.data.csv): ")
    headers_file_name = input("Digite o nome do ARQUIVO DE HEADERS CSV (ex: headers.csv): ")
    delimiter = input("Digite o delimitador do CSV (ex: ',' ou ';'): ")
    target_name = input("Digite o NOME da coluna target para remover (ou deixe vazio): ")
    normalize_choice = input("Normalizar features? (S/N): ").upper()
    
    target_name = target_name if target_name.strip() else None
    perform_normalization = (normalize_choice == 'S')
    
    X, y_true, feature_names = load_and_preprocess_data(data_file_name, headers_file_name, delimiter, target_name, perform_normalization)

    if X is None:
        exit()
        
    try:
        n_clusters_input = int(input("Quantidade de clusters (K): "))
        Tw_input = float(input("Parâmetro de Regularização da Largura Tw (ex: 0.1, 1.0, 10.0): "))
        Tu_input = float(input("Parâmetro de Regularização da Pertinência Tu (ex: 0.1, 1.0): "))
        n_repetitions_input = int(input("Quantidade de repetições: "))
        max_iter_input = int(input("Máximo de iterações por rodada: "))
        
        beta_val = 1.0
        if y_true is not None:
            beta_input = input("Digite o valor de Beta para o F-measure (ex: 1 para F1-score, 0.5 priorizar precisão): ")
            if beta_input.strip():
                beta_val = float(beta_input)
                
    except ValueError:
        print("Entrada inválida.")
        exit()
        
    # --- Execução do KFCM-K-W-EwEu.2 ---
    best_objective_value = float('inf')
    best_s_squared = None
    best_U = None 
    best_G = None 
    best_labels = None
    best_round = None 
    
    TOLERANCE = 1e-6 
    
    # Armazenar histórico de métricas
    all_metrics_history = []
    
    for i in range(n_repetitions_input):
        print(f"\n--- Repetição {i+1}/{n_repetitions_input} (KFCM-K-W-EwEu.2) ---")
        
        kfcm_weweu2 = KernelFCMWEWEU2(n_clusters=n_clusters_input, Tw=Tw_input, Tu=Tu_input, tol=TOLERANCE, max_iter=max_iter_input, random_state=i) 
        
        initial_prototypes, final_labels, final_U, final_G, final_s_squared, current_objective_value = kfcm_weweu2.fit(X)
        
        # CÁLCULO DE MÉTRICAS POR RODADA
        if y_true is not None:
            round_metrics = compute_evaluation_metrics(y_true, final_labels, final_U, n_clusters_input, beta_val)
            round_metrics['Round'] = i + 1
            round_metrics['J_value'] = current_objective_value
            all_metrics_history.append(round_metrics)

        print("\n--- Resultados da Rodada ---")
        print(f"Valor final da função J_KFCM-K-W-EwEu.2: {current_objective_value:.6f}")
        
        if current_objective_value < best_objective_value:
            best_objective_value = current_objective_value
            best_U = final_U
            best_G = final_G
            best_labels = final_labels
            best_s_squared = final_s_squared
            best_round = i + 1

    print("\n\n" + "="*60)
    print("RESULTADOS GERAIS")
    print("="*60)
    print(f"\nMenor J encontrado: {best_objective_value:.6f} (Rodada {best_round})")

    print("\n--- Parâmetros de Largura Finais (s^2) por Cluster ---")
    row_indices = [f'Cluster {c}' for c in range(n_clusters_input)]
    df_s_squared = pd.DataFrame(best_s_squared, index=row_indices, columns=feature_names)
    print(df_s_squared)

    # --- Salvamento de Dados do Melhor Resultado ---
    df_G = pd.DataFrame(best_G, columns=feature_names)
    df_G.to_csv("EwEu.2_prototipos_finais_otimos.csv", index_label="cluster")
    
    df_s_squared.to_csv("EwEu.2_s_squared_otimos.csv", index_label="cluster")
    
    pd.DataFrame(best_U).to_csv("EwEu.2_pertinencia_fuzzy_otima.csv", index=False)

    df_results = pd.DataFrame(X, columns=feature_names)
    df_results['cluster'] = best_labels
    if y_true is not None:
        df_results['original_target'] = y_true.values 
        
        best_cm_df = generate_confusion_matrix_clustering(y_true.values, best_labels, n_clusters_input)
        best_cm_df.to_csv("EwEu.2_matriz_de_confusao_otima.csv", index=True)
        
    df_results.to_csv("EwEu.2_resultados_clusters_otimos.csv", index=False)

    print("\nArquivos de parâmetros ótimos (G, s^2, U, Resultados) salvos com sucesso.")

    # -------------------- APRESENTAÇÃO DAS MÉTRICAS GLOBAIS --------------------
    if y_true is not None and len(all_metrics_history) > 0:
        
        print("\n\n--- Matriz de Confusão do Melhor Resultado ---")
        print(best_cm_df)
        
        df_all_metrics = pd.DataFrame(all_metrics_history)
        
        # Organizando colunas para visualização
        cols_order = ['Round', 'J_value'] + [c for c in df_all_metrics.columns if c not in ['Round', 'J_value']]
        df_all_metrics = df_all_metrics[cols_order]

        print("\n\n--- Resumo Estatístico das Métricas ---")
        cols_to_agg = [col for col in df_all_metrics.columns if col not in ['Round']]
        
        stats_df = df_all_metrics[cols_to_agg].agg(['mean', 'std', 'median', 'max', 'min']).T
        stats_df = stats_df.rename(columns={
            'mean': 'Média', 
            'std': 'Desvio Padrão', 
            'median': 'Mediana', 
            'max': 'Máximo', 
            'min': 'Mínimo'
        })
        print(stats_df)
        
        stats_df.to_csv("EwEu.2_metricas_estatisticas_gerais.csv", index_label="Metrica")
        print("\nEstatísticas (Média, Desvio, Mediana, Máx, Mín) salvas em 'EwEu.2_metricas_estatisticas_gerais.csv'")

        print(f"\n--- Métricas da Melhor Execução (Rodada {best_round}) ---")
        best_metrics = df_all_metrics[df_all_metrics['Round'] == best_round].iloc[0]
        print(best_metrics)
        
        best_metrics.to_frame().T.to_csv("EwEu.2_metricas_melhor_rodada.csv", index=False)
        df_all_metrics.to_csv("EwEu.2_historico_metricas_todas_rodadas.csv", index=False)
        print("Histórico completo e métricas da melhor rodada salvos em CSV.")
        
    else:
        print("\n\n--- Métricas ---")
        print("Aviso: As métricas estatísticas não foram calculadas pois o target real não foi fornecido.")
