# synpick

Preparation
-----------

```bash
python setup.py develop

cd external_data

# Meshes
wget http://ptak.felk.cvut.cz/6DB/public/bop_datasets/ycbv_models.zip
unzip ycbv_models.zip

# IBL files (lighting environment)
wget http://www.hdrlabs.com/sibl/archive/downloads/Chiricahua_Plaza.zip # or any other
unzip Chiricahua_Plaza.zip
```

Running
-------

```bash
python synpick/generate_moving.py --ibl external_data/ibl/Siggraph2007_UpperFloor/Siggraph2007_UpperFloor.ibl --out output/synpick/train --base 0
```
