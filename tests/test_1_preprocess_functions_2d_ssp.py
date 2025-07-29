def test_preprocess_data_synthetic(tmp_path):
    import xarray as xr
    import numpy as np
    from itwinai.plugins.xtclim.preprocessing.preprocess_functions_2d_ssp import PreprocessData

    # Créer les dossiers
    input_dir = tmp_path / "input_dir"
    input_dir.mkdir()

    dataset_root = tmp_path / "dataset_root"
    dataset_root.mkdir()

    # Fichier NetCDF d'entrée (historique)
    data = xr.DataArray(np.random.rand(3, 2, 2), dims=["time", "lat", "lon"],
                        coords={"time": [0, 1, 2], "lat": [0.25, 0.75], "lon": [0.25, 0.75]})
    ds = xr.Dataset({"tas": data})
    input_file = input_dir / "input.nc"
    ds.to_netcdf(input_file)

    # Fichier land‐sea mask requis dans dataset_root
    mask = dataset_root / "landsea_mask.nc"
    mask_ds = ds["tas"].isel(time=0).to_dataset(name="mask")
    mask_ds.attrs["variable_id"] = "mask"
    mask_ds.to_netcdf(mask)

    # Fichier SSP dans dataset_root (on donne le nom exact "ssp1.nc")
    ssp_file = dataset_root / "ssp1.nc"
    ds.attrs["variable_id"] = "tas"
    ds.to_netcdf(ssp_file)

    # Construire le dict scenario_extr attendu par PreprocessData
    #  - clé "ssp1" avec liste de fichiers (nom relatif à dataset_root)
    scenario_extr = {"ssp1": ["ssp1.nc"]}

    # Exécuter l’étape : on ne touche pas au code source,
    # on passe uniquement le paramètre scenario_extr sous forme de dict
    step = PreprocessData(
        dataset_root=str(dataset_root),
        input_path=str(input_dir),
        output_path=str(tmp_path),
        histo_extr=str(input_file),         # historiquement non utilisé dans execute()
        landsea_mask="landsea_mask.nc",     # fichier mask dans dataset_root
        min_lon=0,
        max_lon=1,
        min_lat=0,
        max_lat=1,
        scenarios=["ssp1"],                  # liste de scénarios (ici un seul "ssp1")
        scenario_extr=scenario_extr,         # dict mapping "ssp1" -> ["ssp1.nc"]
    )

    step.execute()

    # Vérifier que les fichiers prétraités sont bien créés dans input_dir
    assert (input_dir / "preprocessed_2d_train_data_allssp.npy").exists()
    assert (input_dir / "preprocessed_2d_test_data_allssp.npy").exists()
    assert (input_dir / "dates_train_data.csv").exists()
    assert (input_dir / "dates_test_data.csv").exists()
    # Et pour le scénario "ssp1", on doit également avoir :
    assert (input_dir / "preprocessed_2d_projssp1_data_allssp.npy").exists()
    assert (input_dir / "dates_projssp1_data.csv").exists()
