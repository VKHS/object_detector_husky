# yolo_gnn_refiner

object detector husky - Ett ROS2-paket för förbättring av bounding boxes från YOLO-detektioner med hjälp av Graph Neural Networks (GNN).

## Beskrivning

Detta paket implementerar en metod för att förbättra YOLO-detektioner genom att använda en Graph Neural Network (GNN). Systemet:

1. Laddar en förtränad YOLOv8-modell och fryser den
2. För varje bild får den YOLO-förslag (boxes/scores/labels)
3. Bygger en graf över förslagen (noder=boxes, viktade kanter baserat på IoU+avstånd, plus kNN)
4. Använder en liten GNN för att förutsäga per-box delta (dx1,dy1,dx2,dy2), uppdaterar boxes => förfinade boxes
5. Tränar GNN med Smooth L1 till ground-truth (COCO YOLO-txt), rapporterar valideringsförlust och genomsnittlig IoU-förbättring
6. Sparar bästa checkpoint och kör förfinad detektion över en testuppsättning

## Krav

- ROS2 (testat med Jazzy)
- Python 3.8+
- PyTorch
- Ultralytics YOLO
- PyTorch Geometric
- OpenCV
- NumPy

## Installation

### Bygga paketet

För att bygga paketet med colcon:

```bash
cd /home/emil/husky_sim/husky_ws
colcon build --packages-select object_detector_husky
```

För att bygga med symlink-install (för utveckling):

```bash
colcon build --symlink-install --packages-select object_detector_husky
```

Efter bygget, källan till workspace:

```bash
source install/setup.bash
```

## Användning

### Köra med ROS2 launch

För att köra object_detector_husky med ROS2 launch:

```bash
ros2 launch object_detector_husky object_detector_husky.launch.py
```

Du kan också ange argument för att anpassa körningen:

```bash
ros2 launch object_detector_husky object_detector_husky.launch.py \
  mode:=train_detect \
  model:=weights.pt \
  train_dir:=/path/to/train/images \
  train_annot:=/path/to/train/annotations \
  test_dir:=/path/to/test/images \
  out_dir:=/path/to/output \
  epochs:=5 \
  device:=cuda
```

### Tillgängliga launch-argument

- `mode`: Körläge - `train`, `detect`, eller `train_detect` (standard: `train_detect`)
- `model`: Sökväg till modell (.pt-fil) (standard: `weights.pt`)
- `train_dir`: Sökväg till träningsbilder
- `train_annot`: Sökväg till träningsannoteringar
- `test_dir`: Sökväg till testbilder
- `out_dir`: Utdatamapp för resultat
- `epochs`: Antal träningsepoker (standard: `5`)
- `device`: Enhet att köra på - `cuda` eller `cpu` (standard: `cuda`)

### Köra direkt (utan ROS2 launch)

Du kan också köra skriptet direkt:

```bash
ros2 run object_detector_husky object_detector_husky --mode train_detect --model weights.pt [andra argument...]
```

## Projektstruktur

```
object_detector_husky/
├── CMakeLists.txt          # CMake-konfiguration
├── package.xml             # ROS2 paketdefinition
├── setup.py                # Python package setup
├── setup.cfg               # Setuptools konfiguration
├── README.md               # Denna fil
├── launch/                 # Launch-filer
│   └──object_detector_husky.launch.py
├── object_detector_husky/       # Python-paket
│   ├── __init__.py
│   └── object_detector_husky.py # Huvudskript
├── weights/                # YOLO-vikter
│   └── weights.pt
├── resource/               # ROS2 resource-fil
│   └── object_detector_husky
└── test/                   # Test-filer
    ├── test_copyright.py
    ├── test_flake8.py
    └── test_pep257.py
```

## Ytterligare argument

För en fullständig lista över alla tillgängliga kommandoradsargument, kör:

```bash
ros2 run object_detector_husky object_detector_husky --help
```

## Noteringar

- Node-funktioner: [geometri (normaliserad) + YOLO-score + träningsbar klass-embedding] → MLP → GNN
- Grafkanter: vikt = alpha*IoU + (1-alpha)*exp(-dist/sigma), plus kNN-grannar för att säkerställa anslutning
- Endast GNN (och liten feature MLP) tränas. YOLO är fryst genom hela processen

## Byggning och utveckling

Paketet är konfigurerat för att fungera med både vanlig `colcon build` och `colcon build --symlink-install` för utveckling. Symlink-install gör att ändringar i källkoden omedelbart reflekteras utan att behöva bygga om paketet.

## Licens

TODO: Licensdeklaration

## Författare

vahab
