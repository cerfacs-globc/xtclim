import numpy as np
import pandas as pd

from itwinai.plugins.xtclim.preprocessing.preprocess_2d_seasons import SplitPreprocessedData


def test_split_preprocessed_data(tmp_path):
    # === Créer le dossier d'entrée
    input_dir = tmp_path
    input_path = str(input_dir)
    n_memb = 1
    scenarios = ["126"]

    # === Données simulées : 3 années = 3 * 365 = 1095 jours
    n_days = 3 * 365
    height, width = 2, 2  # image shape

    # === Données train/test/proj (prétraitées)
    train_images = np.random.rand(n_days, height, width).astype("float32")
    test_images = np.random.rand(n_days, height, width).astype("float32")
    proj_images = np.random.rand(n_days, height, width).astype("float32")

    np.save(input_dir / "preprocessed_2d_train_data_allssp.npy", train_images)
    np.save(input_dir / "preprocessed_2d_test_data_allssp.npy", test_images)
    np.save(input_dir / "preprocessed_2d_proj126_data_allssp.npy", proj_images)

    # === Dates : on génère une DataFrame avec des dates journalières
    base_dates = pd.date_range("2000-01-01", periods=n_days)
    train_time = pd.DataFrame({"index": range(n_days), "date": base_dates})
    test_time = pd.DataFrame({"index": range(n_days), "date": base_dates})
    proj_time = pd.DataFrame({"index": range(n_days), "date": base_dates})

    train_time.to_csv(input_dir / "dates_train_data.csv", index=False)
    test_time.to_csv(input_dir / "dates_test_data.csv", index=False)
    proj_time.to_csv(input_dir / "dates_proj126_data.csv", index=False)

    # === Exécuter l'étape
    step = SplitPreprocessedData(input_path=input_path, scenarios=scenarios, n_memb=n_memb)
    step.execute()

    # === Vérifier que les fichiers ont été créés
    for dataset_type in ["train", "test", "proj126"]:
        for season in ["winter", "spring", "summer", "autumn"]:
            npy_file = input_dir / f"preprocessed_1d_{dataset_type}_{season}_data_{n_memb}memb.npy"
            assert npy_file.exists(), f"Manquant : {npy_file}"

    for dataset_type in ["train", "test", "proj"]:
        for season in ["winter", "spring", "summer", "autumn"]:
            csv_file = input_dir / f"dates_{dataset_type}_{season}_data_{n_memb}memb.csv"
            assert csv_file.exists(), f"Manquant : {csv_file}"
