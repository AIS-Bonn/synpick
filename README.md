# synpick

Preparation
-----------

```bash
python setup.py develop

mkdir external_data
cd external_data

# Meshes
mkdir ycbv_models
cd ycbv_models

wget http://ptak.felk.cvut.cz/6DB/public/bop_datasets/ycbv_models.zip
unzip ycbv_models.zip

cd ..

# IBL files (lighting environment)
mkdir ibl
cd ibl

wget http://www.hdrlabs.com/sibl/archive/downloads/Chiricahua_Plaza.zip # or any other
unzip Chiricahua_Plaza.zip
```

Running
-------

```bash
python synpick/generate_moving.py --ibl external_data/ibl/Chiricahua_Plaza/Chiricahua_Plaza.ibl --out output/synpick/train --base 0
```

or

```bash
python synpick/generate_picking.py --ibl external_data/ibl/Chiricahua_Plaza/Chiricahua_Plaza.ibl --out output/synpick/train --base 0
```

