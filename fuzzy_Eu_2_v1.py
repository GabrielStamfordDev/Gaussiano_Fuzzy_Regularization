import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler 
import sys 
import math

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

def gaussian_kernel(X_A, X_B, s_squared):
    """
    Calcula a Matriz Kernel Local K^(s_i)(X_A, X_B).
    Para o modelo EU.2, s_squared é uma matriz (K_clusters, D).
    """
    s_squared = np.asarray(s_squared)
    inv_s_squared = np.zeros_like(s_squared)
    non_zero_mask = s_squared > 1e-15
    inv_s_squared[non_zero_mask] = 1.0 / s_squared[non_zero_mask]
    
    diff = X_A[:, np.newaxis, :] - X_B[np.newaxis, :, :] 
    diff_sq = diff**2 
    # Adicionamos um eixo [np.newaxis, :, :] para expandir as N amostras, 
    # multiplicando corretamente cada cluster pelas suas larguras locais
    weighted_diff_sq = diff_sq * inv_s_squared[np.newaxis, :, :]
    sum_weighted_diff_sq = np.sum(weighted_diff_sq, axis=2) 
    K = np.exp(-0.5 * sum_weighted_diff_sq)
    return K

# -------------------- CLASSE KFCM-K-W-EU.2 --------------------

class KernelFCMWEU2:
    """
    Implementa o KFCM-K-W-EU.2: Kernel Fuzzy C-Means com cálculo automático de 
    larguras LOCAIS por cluster (restrição de produto) e regularização de pertinência (Tu).
    """
    def __init__(self, n_clusters=3, Tu=0.1, tol=1e-6, max_iter=100, random_state=None):
        self.n_clusters = n_clusters
        self.Tu = Tu 
        self.tol = tol
        self.max_iter = max_iter 
        self.random_state = random_state
        self.G = None 
        self.U = None 
        self.s_squared = None 
        
    def get_labels(self):
        if self.U is None:
            return None
        return np.argmax(self.U, axis=1)

    # Função Objetivo J_KFCM-K-W-EU.2 (Eq. 15)
    def calculate_objective_function(self, X, G, U, s_squared):
        K_XG = gaussian_kernel(X, G, s_squared) 
        D_XG = 2 * (1 - K_XG)
        term1 = np.sum(U * D_XG)
        
        term_entropy_u = (1 + U) * np.log(1 + U)
        term2 = self.Tu * np.sum(term_entropy_u)
        
        return term1 + term2

    # Step 3: Atualiza U (Algoritmo 1 / Eq. 27)
    def update_U(self, X, G, s_squared):
        N, D = X.shape
        K_clusters = self.n_clusters
        
        K_XG = gaussian_kernel(X, G, s_squared)
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

    # Step 2: Atualiza G (Eq. 21)
    def update_G(self, X, U, s_squared):
        K_clusters = self.n_clusters
        K_XG = gaussian_kernel(X, self.G, s_squared) 
        new_G = np.zeros_like(self.G)
        
        for i in range(K_clusters):
            W_i = U[:, i] * K_XG[:, i] 
            numerator = np.sum(W_i[:, np.newaxis] * X, axis=0)
            denominator = np.sum(W_i)
            
            if denominator < 1e-9:
                new_G[i] = self.G[i].copy()
            else:
                new_G[i] = numerator / denominator

        return new_G

    # Step 1: Atualiza s_squared usando a restrição de produto local (Eq. 19)
    def update_s_squared(self, X, G, U):
        N, D = X.shape
        K_clusters = self.n_clusters
        K_XG = gaussian_kernel(X, G, self.s_squared)
        
        # Q agora é uma matriz de penalidades (K_clusters, D)
        Q = np.zeros((K_clusters, D))
        for i in range(K_clusters):
            for j in range(D):
                diff_sq_ij = (X[:, j] - G[i, j])**2 
                W_ki = U[:, i] * K_XG[:, i]
                Q[i, j] = np.sum(W_ki * diff_sq_ij)
            
        log_Q = np.log(Q + 1e-15) 
        # O numerador é calculado localmente para cada cluster i
        numerator = np.exp(np.sum(log_Q, axis=1) / D) # Shape: (K_clusters,)
        
        # Eq 19: 1/s_ij^2 = Numerador_i / Q_ij
        new_inv_s_squared = numerator[:, np.newaxis] / (Q + 1e-15)
        
        new_s_squared = np.zeros((K_clusters, D))
        for i in range(K_clusters):
            for j in range(D):
                if new_inv_s_squared[i, j] > 1e-15:
                    new_s_squared[i, j] = 1.0 / new_inv_s_squared[i, j]
                else:
                    new_s_squared[i, j] = 1e15 # Infinito prático
                
        return new_s_squared

    def fit(self, X):
        np.random.seed(self.random_state)
        N, D = X.shape
        K_clusters = self.n_clusters
        
        random_idx = np.random.choice(N, K_clusters, replace=False)
        self.G = X[random_idx].copy() 
        initial_prototypes = self.G.copy()

        # Inicialização com prod(1/s_ij^2) = 1 -> Inicializamos com matriz 1.0
        self.s_squared = np.ones((K_clusters, D))
        
        self.U = self.update_U(X, self.G, self.s_squared)
        
        prev_objective_value = self.calculate_objective_function(X, self.G, self.U, self.s_squared)
        print(f"  Iteração 0: J_NEW = {prev_objective_value:.6f}")
        
        iteration_count = 1
        
        while(iteration_count <= self.max_iter):
            J_OLD = prev_objective_value
            
            self.s_squared = self.update_s_squared(X, self.G, self.U)
            self.G = self.update_G(X, self.U, self.s_squared)
            self.U = self.update_U(X, self.G, self.s_squared)
            
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
    
    print("--- Configuração do Dataset (KFCM-K-W-EU.2) ---")
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
        Tu_input = float(input("Parâmetro de Regularização da Pertinência Tu (ex: 0.1, 1.0): "))
        n_repetitions_input = int(input("Quantidade de repetições: "))
        max_iter_input = int(input("Máximo de iterações por rodada: "))
    except ValueError:
        print("Entrada inválida.")
        exit()
        
    best_objective_value = float('inf')
    best_s_squared = None
    best_U = None 
    best_G = None 
    best_labels = None
    best_round = None 
    
    TOLERANCE = 1e-6 
    
    for i in range(n_repetitions_input):
        print(f"\n--- Repetição {i+1}/{n_repetitions_input} (KFCM-K-W-EU.2) ---")
        
        kfcm_weu2 = KernelFCMWEU2(n_clusters=n_clusters_input, Tu=Tu_input, tol=TOLERANCE, max_iter=max_iter_input, random_state=i) 
        
        initial_prototypes, final_labels, final_U, final_G, final_s_squared, current_objective_value = kfcm_weu2.fit(X)
        
        print("\n--- Resultados da Rodada ---")
        print(f"Protótipos Iniciais:\n{initial_prototypes}")
        print(f"\nValor final da função J_KFCM-K-W-EU.2: {current_objective_value:.6f}")
        
        print("\n--- Afiliação dos Clusters (Rótulos Hard) ---")
        for j in range(n_clusters_input):
            cluster_indices = np.where(final_labels == j)[0]
            print(f"  Cluster C{j} ({len(cluster_indices)} pontos): {[int(x) for x in cluster_indices]}")
            
        print("\n--- Coordenadas Finais dos Protótipos (Matriz G) ---")
        print(final_G)

        if y_true is not None:
            print("\n\n--- Matriz de Confusão (Clusters vs. Classes Verdadeiras) ---")
            cm_df = generate_confusion_matrix_clustering(y_true.values, final_labels, n_clusters_input)
            print(cm_df)
        else:
            print("\n\n--- Matriz de Confusão ---")
            print("Não foi possível gerar a Matriz de Confusão, pois a coluna de rótulos/target não foi fornecida.")
        
        if current_objective_value < best_objective_value:
            best_objective_value = current_objective_value
            best_U = final_U
            best_G = final_G
            best_labels = final_labels
            best_s_squared = final_s_squared
            best_round = i + 1

    print("\n\n--- Resultado Final (Melhor Execução - KFCM-K-W-EU.2) ---")
    print(f"Menor J encontrado: {best_objective_value:.6f} (Rodada {best_round})")

    print("\n--- Parâmetros de Largura Finais (s^2) ---")
    # Alterado para formatar a matriz 2D corretamente
    df_s_squared = pd.DataFrame(best_s_squared, columns=feature_names, index=[f"Cluster {i}" for i in range(n_clusters_input)])
    print(df_s_squared)

    print("\n--- Afiliação dos Clusters (Rótulos Hard) ---")
    for j in range(n_clusters_input):
        cluster_indices = np.where(best_labels == j)[0]
        print(f"  Cluster C{j} ({len(cluster_indices)} pontos): {[int(x) for x in cluster_indices]}")

     # --- 4. Cálculo, Print e Salvamento da Matriz de Confusão do Melhor Resultado ---
    print("\n\n--- Matriz de Confusão do Melhor Resultado (Clusters vs. Classes Verdadeiras) ---")

    if y_true is not None:
        best_cm_df = generate_confusion_matrix_clustering(y_true.values, best_labels, n_clusters_input)
        print(best_cm_df)
        best_cm_df.to_csv("matriz_de_confusao_otima.csv", index=True)
        print("\nMatriz de Confusão ótima salva em 'matriz_de_confusao_otima.csv'")
    else:
        print("Não foi possível gerar a Matriz de Confusão, pois a coluna de rótulos/target não foi fornecida.")

    print("\n--- Matriz de Pertinência Fuzzy (U) final ---")
    for i in range(best_U.shape[0]):
        row_str = "["
        for j in range(best_U.shape[1]):
            if j > 0:
                row_str += " "
            row_str += f"{best_U[i,j]:.6f}"
        row_str += "]"
        print(row_str)
    
    print("\n--- Salvando Resultados em Arquivos CSV ---")
    
    df_G = pd.DataFrame(best_G, columns=feature_names)
    df_G.to_csv("prototipos_finais_kfcm-weu2_otimos.csv", index_label="cluster")
    print("Matriz de Protótipos (G) ótima salva em 'prototipos_finais_kfcm-weu2_otimos.csv'")
    
    # Salva a matriz 2D de larguras locais
    df_s_squared.to_csv("s_squared_kfcm-weu2_otimos.csv", index_label="cluster")
    print("Matriz de Parâmetros de Largura Locais (s^2) salva em 's_squared_kfcm-weu2_otimos.csv'")
    
    df_U = pd.DataFrame(best_U)
    df_U.to_csv("matriz_de_pertencimento_fuzzy_kfcm-weu2_otima.csv", index=False)
    print("Matriz de Pertinência Fuzzy (U) ótima salva em 'matriz_de_pertencimento_fuzzy_kfcm-weu2_otima.csv'")

    df_results = pd.DataFrame(X, columns=feature_names)
    df_results['cluster'] = best_labels
    if y_true is not None:
        df_results['original_target'] = y_true.values 
    df_results.to_csv("resultados_com_clusters_kfcm-weu2_otimos.csv", index=False)
    print("Dados originais com os rótulos salvos em 'resultados_com_clusters_kfcm-weu2_otimos.csv'")

    # -------------------- CÁLCULO DE MÉTRICAS EXTERNAS --------------------

    if y_true is not None:
        from sklearn.metrics import adjusted_rand_score, accuracy_score, fbeta_score, normalized_mutual_info_score
        import math

        print("\n\n--- Cálculo de Métricas de Avaliação ---")
        
        ari = adjusted_rand_score(y_true, best_labels)
        print(f"ARI (Adjusted Rand Index): {ari:.4f}")

        nmi = normalized_mutual_info_score(y_true, best_labels)
        print(f"NMI (Normalized Mutual Information): {nmi:.4f}")

        y_true_np = y_true.values
        n_samples = len(y_true)
        unique_classes = np.unique(y_true_np)
        
        entropy_conditional = 0
        for k in range(n_clusters_input):
            mask = (best_labels == k)
            cluster_size = np.sum(mask)
            
            if cluster_size > 0:
                true_labels_in_cluster = y_true_np[mask]
                cluster_impurity = 0
                for cls in unique_classes:
                    p_ij = np.sum(true_labels_in_cluster == cls) / cluster_size
                    if p_ij > 0:
                        cluster_impurity -= p_ij * math.log2(p_ij)
                
                entropy_conditional += (cluster_size / n_samples) * cluster_impurity
                
        print(f"\nEntropia Condicional (Impureza - Quanto menor, melhor): {entropy_conditional:.4f}")

        try:
            print("\nConfiguração do F-measure:")
            beta_input = input("Digite o valor de Beta (ex: 1 para F1-score, 0.5 para priorizar precisão, 2 para recall): ")
            beta_val = float(beta_input)
        except ValueError:
            print("Entrada inválida. Assumindo Beta = 1.0 (F1-score padrão).")
            beta_val = 1.0

        y_true_np = y_true.values
        predicted_labels_mapped = np.empty_like(best_labels, dtype=object) 
        cluster_class_map = {}

        for k in range(n_clusters_input):
            mask = (best_labels == k)
            if np.sum(mask) > 0:
                true_labels_in_cluster = y_true_np[mask]
                most_frequent_class = pd.Series(true_labels_in_cluster).mode()[0]
                predicted_labels_mapped[mask] = most_frequent_class
                cluster_class_map[k] = most_frequent_class
            else:
                pass
        
        if pd.api.types.is_numeric_dtype(y_true):
            predicted_labels_mapped = predicted_labels_mapped.astype(y_true.dtype)

        print(f"\nMapeamento automático (Cluster -> Classe Predominante): {cluster_class_map}")

        acc = accuracy_score(y_true_np, predicted_labels_mapped)
        print(f"Acurácia (Baseada na classe majoritária do cluster): {acc:.4f}")

        ER = 1 - acc
        print(f"Error of Classification (ER): {ER:.4f}")

        f_meas_weighted = fbeta_score(y_true_np, predicted_labels_mapped, beta=beta_val, average='weighted', zero_division=0)
        print(f"F-measure Global (Média Ponderada, Beta={beta_val}): {f_meas_weighted:.4f}")

        f_meas_per_class = fbeta_score(y_true_np, predicted_labels_mapped, beta=beta_val, average=None, zero_division=0)
        
        unique_classes = np.unique(y_true_np)
        
        print(f"\n--- F-measure Detalhado por Classe (Beta={beta_val}) ---")
        per_class_data = []
        for cls, score in zip(unique_classes, f_meas_per_class):
            print(f"  Classe '{cls}': {score:.4f}")
            per_class_data.append({'Classe': cls, 'F-measure': score})

        summary_data = [
            {'Metrica': 'ARI', 'Valor': ari, 'Detalhe': 'Global (Pares)'},
            {'Metrica': 'NMI', 'Valor': nmi, 'Detalhe': 'Global (Informação)'},
            {'Metrica': 'Acuracia', 'Valor': acc, 'Detalhe': 'Global (Purity)'},
            {'Metrica': 'Error (ER)', 'Valor': ER, 'Detalhe': 'Global (1 - Acc)'},
            {'Metrica': f'F-measure (Weighted)', 'Valor': f_meas_weighted, 'Detalhe': f'Beta={beta_val}'},
            {'Metrica': 'Entropia Condicional (Impureza)', 'Valor': entropy_conditional}
        ]
        
        for item in per_class_data:
            summary_data.append({
                'Metrica': f"F-measure (Classe {item['Classe']})",
                'Valor': item['F-measure'],
                'Detalhe': f'Beta={beta_val}'
            })

        metrics_df = pd.DataFrame(summary_data)
        metrics_df.to_csv("metricas_avaliacao_detalhadas.csv", index=False)
        print("\nArquivo 'metricas_avaliacao_detalhadas.csv' salvo com sucesso.")

    else:
        print("\n\n--- Métricas ---")
        print("Aviso: As métricas não foram calculadas pois o target real não foi fornecido.")

# -------------------- MÉTRICAS FUZZY BASEADAS EM PARES --------------------
        
    print("\n--- Métricas Fuzzy Baseadas em Pares (Eqs. 38-45 com pertinência contínua e Rand Index) ---")
    
    try:
        y_np_flat = np.asarray(y_true).ravel()
        N_samples = len(y_np_flat)
        
        Psi_fuzzy = np.dot(best_U, best_U.T)

        Psi_true = np.zeros((N_samples, N_samples))
        for i in range(N_samples):
            for j in range(N_samples):
                if y_np_flat[i] == y_np_flat[j]:
                    Psi_true[i, j] = 1.0

        rows, cols = np.triu_indices(N_samples, k=1)
        
        vec_fuzzy = Psi_fuzzy[rows, cols]
        vec_true = Psi_true[rows, cols]

        n_ss_fuzzy = np.sum(vec_fuzzy * vec_true)
        n_sd_fuzzy = np.sum(vec_fuzzy * (1 - vec_true))
        n_ds_fuzzy = np.sum((1 - vec_fuzzy) * vec_true)
        n_dd_fuzzy = np.sum((1 - vec_fuzzy) * (1 - vec_true))

        denom_jaccard = n_ss_fuzzy + n_sd_fuzzy + n_ds_fuzzy
        jaccard_fuzzy_score = n_ss_fuzzy / denom_jaccard if denom_jaccard > 0 else 0.0
        
        print(f"Fuzzy Jaccard Coefficient: {jaccard_fuzzy_score:.4f}")
        print(f"  Componentes -> N_SS: {n_ss_fuzzy:.2f}, N_SD: {n_sd_fuzzy:.2f}, N_DS: {n_ds_fuzzy:.2f}")

        denom_fm = math.sqrt((n_ss_fuzzy + n_sd_fuzzy) * (n_ss_fuzzy + n_ds_fuzzy))
        fm_fuzzy_score = n_ss_fuzzy / denom_fm if denom_fm > 0 else 0.0
        
        print(f"Fuzzy Folkes-Mallows Index: {fm_fuzzy_score:.4f}")

        total_pairs = n_ss_fuzzy + n_sd_fuzzy + n_ds_fuzzy + n_dd_fuzzy
        rand_frigui = (n_ss_fuzzy + n_dd_fuzzy) / total_pairs if total_pairs > 0 else 0.0
        
        print(f"Fuzzy Rand Index (Frigui): {rand_frigui:.4f}")
        print(f"  Componente N_DD (Concordância Negativa): {n_dd_fuzzy:.2f}")

        fuzzy_metrics_data = [
            {'Metrica': 'Fuzzy Jaccard', 'Valor': jaccard_fuzzy_score, 'Detalhe': 'Baseado em U contínuo'},
            {'Metrica': 'Fuzzy Folkes-Mallows', 'Valor': fm_fuzzy_score, 'Detalhe': 'Baseado em U contínuo'},
            {'Metrica': 'Fuzzy Rand Index (Frigui)', 'Valor': rand_frigui, 'Detalhe': 'Inclui N_DD'}
        ]
        
        if 'metrics_df' in locals():
            current_data = metrics_df.to_dict('records')
            current_data.extend(fuzzy_metrics_data)
            metrics_df_final = pd.DataFrame(current_data)
        else:
            metrics_df_final = pd.DataFrame(fuzzy_metrics_data)
            
        metrics_df_final.to_csv("metricas_avaliacao_fuzzy_completas.csv", index=False)
        print("\nArquivo 'metricas_avaliacao_fuzzy_completas.csv' salvo com sucesso.")

    except Exception as e:
        print(f"Erro ao calcular métricas fuzzy: {e}")