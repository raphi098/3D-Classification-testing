import pandas as pd

# Data for the table
data = {
    "File": [
        "1gliedrig_flach_(10038).stl",
        "1gliedrig_flach_(10039).stl",
        "1gliedrig_glatt_(15680).stl",
        "1gliedrig_glatt_(15681).stl",
        "1gliedrig_glatt_(6943).stl",
        "1gliedrig_glatt_(6944).stl",
        "1gliedrig_kappe_front_(136).stl",
        "1gliedrig_sattel_(1038).stl",
        "1gliedrig_sattel_(1040).stl",
        "1gliedrig_sattel_(2551).stl",
        "1gliedrig_stufe_(1037).stl",
        "1gliedrig_stufe_(1041).stl"
    ],
    "Prediction": [
        "1gliedrig_sattel",
        "1gliedrig_sattel",
        "1gliedrig_sattel",
        "1gliedrig_sattel",
        "modell_einzelteil_2",
        "1gliedrig_sattel",
        "1gliedrig_sattel",
        "1gliedrig_kappe_front",
        "1gliedrig_stufe",
        "1gliedrig_rundung",
        "1gliedrig_sattel",
        "1gliedrig_glatt"
    ],
    "Description": [
        "Modell is nicht flach",
        "Falsche Vorhersage",
        "Falsche Vorhersage",
        "Falsche Vorhersage",
        "Falsche Vorhersage",
        "Falsche Vorhersage",
        "Falsche Vorhersage",
        "Modell ist eher eine kappe",
        "Modell hat auch eine stufe",
        "Falsche Vorhersage",
        "Falsche Vorhersage",
        "Falsche Vorhersage"
    ]

        }

# Create the DataFrame
df = pd.DataFrame(data)

print(df)