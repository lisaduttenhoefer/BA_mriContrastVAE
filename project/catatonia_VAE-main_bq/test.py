import pandas as pd
# .h5-Datei einlesen (HDF5 Format mit MultiIndex-Spalten)
df = pd.read_hdf("/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data/test_xml_data/Aggregated_cobra.h5")  # ggf. Schlüssel mit `key='...'` angeben

# Nur Spalten mit Level-1-Name == "Vgm" behalten
df_vgm = df.loc[:, df.columns.get_level_values(1) == "Vgm"]

# Optional: nur die erste Ebene der Spalten behalten (z. B. ROI-Namen)
df_vgm.columns = df_vgm.columns.get_level_values(0)

# Ausgabe der Spaltennamen als Liste
column_names = df_vgm.columns.tolist()
#print(column_names)

import pandas as pd


# Index (Zeilennamen) als Liste
row_names = df.index.tolist()

print(len(row_names))