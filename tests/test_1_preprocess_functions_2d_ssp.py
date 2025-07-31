def test_preprocess_data_synthetic(tmp_path):
    import numpy as np
    import xarray as xr

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
    input_file = dataset_root / "input.nc"
    ds.attrs["variable_id"] = "tas"
    ds.to_netcdf(input_file, format="NETCDF3_CLASSIC")

    # Fichier land‐sea mask requis dans dataset_root
    mask = dataset_root / "landsea_mask.nc"
    mask_ds = ds["tas"].isel(time=0).to_dataset(name="mask")
    mask_ds.attrs["variable_id"] = "mask"
    mask_ds.to_netcdf(mask, format="NETCDF3_CLASSIC")

    # Fichier SSP dans dataset_root (on donne le nom exact "585.nc")
    ssp_file = dataset_root / "585.nc"
    ds.attrs["variable_id"] = "tas"
    ds.to_netcdf(ssp_file, format="NETCDF3_CLASSIC")

    # Construire le dict scenario_extr attendu par PreprocessData
    #  - clé "585" avec liste de fichiers (nom relatif à dataset_root)
    scenario_extr = {585: ["585.nc"]}

    # Exécuter l’étape : on ne touche pas au code source,
    # on passe uniquement le paramètre scenario_extr sous forme de dict
    step = PreprocessData(
        dataset_root=str(dataset_root),
        input_path=str(input_dir),
        output_path=str(tmp_path),
        histo_extr=["input.nc"],
        landsea_mask="landsea_mask.nc",     # fichier mask dans dataset_root
        min_lon=0,
        max_lon=1,
        min_lat=0,
        max_lat=1,
        scenarios=[585],                  # liste de scénarios (ici un seul "585")
        scenario_extr=scenario_extr,         # dict mapping "585" -> ["585.nc"]
    )

    step.execute()

    # Vérifier que les fichiers prétraités sont bien créés dans input_dir
    assert (input_dir / "preprocessed_2d_train_data_allssp.npy").exists()
    assert (input_dir / "preprocessed_2d_test_data_allssp.npy").exists()
    assert (input_dir / "dates_train_data.csv").exists()
    assert (input_dir / "dates_test_data.csv").exists()
    # Et pour le scénario "585", on doit également avoir :
    assert (input_dir / "preprocessed_2d_proj585_data_allssp.npy").exists()
    assert (input_dir / "dates_proj585_data.csv").exists()
