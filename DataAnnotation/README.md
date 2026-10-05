# DataAnnotation

Every model in this repository learned from **labeled data**: Fashion-MNIST
classes, Penn-Fudan boxes, Kvasir-SEG masks. Someone drew those labels by
hand. In this folder you become that someone: you label three small datasets,
one for each kind of task you have trained so far, with
[Label Studio](https://labelstud.io), an open-source annotation tool that
runs on your own computer in Docker.

| Folder content | What it is |
| --- | --- |
| [`download_data.py`](download_data.py) | Downloads the three datasets and copies a small, fixed sample of each to `data/annotation/` |
| [`Dockerfile`](Dockerfile) | The Label Studio image for the course (a pinned version, with local file serving turned on) |
| [`compose.yaml`](compose.yaml) | Runs that image: port, storage, your account from `.env`, and the image folder |
| [`label_configs/`](label_configs/) | One labeling interface per task, to paste into a Label Studio project |

There is no notebook in this folder: the work happens in Label Studio, in the
browser.

## Contents

- [Why data annotation matters](#why-data-annotation-matters)
- [Kinds of labels](#kinds-of-labels)
- [Annotation software](#annotation-software)
- [Roboflow, and what it costs](#roboflow-and-what-it-costs)
- [Why we use Label Studio](#why-we-use-label-studio)
- [Letting a model do the first pass: pseudo-labeling and friends](#letting-a-model-do-the-first-pass-pseudo-labeling-and-friends)
- [The three datasets](#the-three-datasets)
- [Step 1: download the data](#step-1-download-the-data)
- [Step 2: run Label Studio with Docker](#step-2-run-label-studio-with-docker)
- [Step 3: label in Label Studio](#step-3-label-in-label-studio)
- [Step 4: export, and check your labels](#step-4-export-and-check-your-labels)
- [Docker cheatsheet](#docker-cheatsheet)
- [Troubleshooting](#troubleshooting)
- [Links](#links)

## Why data annotation matters

- **The labels define the task.** A network learns to reproduce its labels,
  nothing more. If the people who drew the Kvasir-SEG masks had left out the
  polyp edges, the U-Net would learn to leave them out too.
- **Label quality sets the ceiling.** Labels are never perfect: two careful
  people outlining the same polyp agree at a Dice of maybe 0.85 to 0.95, not
  1.0. A model scored against those labels cannot be measured more precisely
  than the labels themselves, so a test Dice of 0.88 may already be close to
  human agreement.
- **Errors in the labels become errors in the model.** A box that is too
  loose, a missed animal in a crowded photo, a cat labeled as a dog: the
  network is trained to make the same mistakes, and the test set rewards it
  for them if the test labels have them too.
- **Consistency matters more than perfection.** Rules such as "box the
  visible part of a partly hidden animal" or "include the balloon's knot,
  not its string" must be written down and followed by everyone. These
  written rules are the **annotation guidelines**, and every serious
  dataset has them.
- **Labeling is expensive.** It is often the largest cost of a project:
  a box takes a few seconds, a careful polygon a minute, a medical mask
  needs an expert. This is why tools, shortcuts, and models that label for
  you (see below) matter.

## Kinds of labels

| Task | One label is | Example in this repo | In this folder |
| --- | --- | --- | --- |
| Image classification | one class per image | Fashion-MNIST, Flowers-102 | dog, church, or parachute |
| Semantic segmentation | a class for every pixel (a mask) | Kvasir-SEG polyps (U-Net) | balloon pixels |
| Object detection | a box and a class for every object | Penn-Fudan pedestrians (YOLO) | buffalo, elephant, rhino, zebra |
| Instance segmentation | a separate mask for every object | (not covered) | one polygon per balloon gives this too |

Other tasks have their own label types: keypoints (body joints), text spans
(named entities), audio transcripts, and time-series events. Label Studio
handles all of them.

## Annotation software

| Tool | Kind | Cost | Good for |
| --- | --- | --- | --- |
| [Label Studio](https://labelstud.io) | Open source (Apache 2.0), web app you run yourself; paid Enterprise and cloud editions | Free (Community edition) | Every data type (images, text, audio, video, time series); configurable interfaces; standard exports |
| [CVAT](https://www.cvat.ai) | Open source (MIT), self-hosted with Docker, or cvat.ai cloud | Free self-hosted; paid cloud plans | Images and especially **video** (object tracking between frames); built-in Segment Anything |
| [Roboflow Annotate](https://roboflow.com/annotate) | Hosted platform: annotate, version, augment, train, deploy | Limited free tier; paid plans | Fast end-to-end computer vision, AI-assisted labeling |
| [makesense.ai](https://www.makesense.ai) | Free website; images never leave your browser | Free | Quick boxes and polygons with no install or account; exports YOLO, VOC, CSV |
| [LabelMe](https://github.com/wkentaro/labelme) | Open source desktop app (Python) | Free | Polygons on a few images, saved as JSON next to each image |
| [VGG Image Annotator (VIA)](https://www.robots.ox.ac.uk/~vgg/software/via/) | A single HTML file that runs offline | Free | Simple polygons and boxes; the balloon dataset was labeled with it |
| [Labelbox](https://labelbox.com), [Encord](https://encord.com), [SuperAnnotate](https://www.superannotate.com), [V7 Darwin](https://www.v7labs.com), [Supervisely](https://supervisely.com) | Commercial platforms | Free tiers with limits; paid per seat or usage | Large teams: workforce management, review queues, quality metrics |
| [Amazon SageMaker Ground Truth](https://aws.amazon.com/sagemaker/groundtruth/) | AWS service with optional paid human labelers | Pay per labeled object | Outsourcing large labeling jobs |
| [doccano](https://github.com/doccano/doccano), [Prodigy](https://prodi.gy) | Text annotation (open source; paid license) | Free; paid | Text classification, named entities |

Pricing and free-tier limits change often; check each tool's website for the
current numbers.

## Roboflow, and what it costs

[Roboflow](https://roboflow.com) is very good at what it does: you upload
images in the browser, draw boxes or click once to get a polygon (its Smart
Polygon uses Segment Anything), keep versions of the dataset, add
augmentations, train a model, and download the dataset in almost any format
with a ready-made code snippet. Many tutorials, including Ultralytics', use
it.

The catch is the price for a class:

- **The free plan is public.** On the free tier, your projects and datasets
  are public on [Roboflow Universe](https://universe.roboflow.com). That is
  fine for balloons, not for medical images, private photos, or a dataset
  you have not published yet.
- **Private data and teams need a paid plan.** Private projects, more images,
  more team members, and more of the usage credits that AI-assisted labeling
  and training consume all come with monthly paid plans
  ([roboflow.com/pricing](https://roboflow.com/pricing)). For a class where
  every student or team needs a private workspace, that adds up quickly.
- **Your data lives on their servers.** Versions, augmentations, and
  trained models are tied to the platform. You can always export, but the
  workflow (and the habit) stays there.

Roboflow is worth knowing and is a good choice for a quick prototype or a
public dataset. For this course we want something free for everyone,
private, and transparent.

## Why we use Label Studio

1. **Free and open source.** The Community edition has no image limits, no
   seat limits, and no credits; the source code is on
   [GitHub](https://github.com/HumanSignal/label-studio).
2. **Your data stays on your computer.** Images are read from a folder on
   your machine and labels are saved locally, which matters for medical,
   personal, or unpublished data.
3. **One tool for every task.** The same app labels images, text, audio,
   video, and time series, so it stays useful in your later projects (for
   example, text for an NLP model).
4. **You see every step.** You write the labeling interface (a short XML
   file, see [`label_configs/`](label_configs/)), export standard formats
   (YOLO, COCO, Pascal VOC, CSV, JSON, mask PNGs), and load them with your
   own code, as the projects in this repository do with their datasets.
5. **Models can help.** Label Studio's
   [ML backend](https://labelstud.io/guide/ml) connects a model (for example
   Segment Anything or YOLO) that pre-labels images for you to correct: the
   techniques of the next section.
6. **Docker makes it identical for everyone.** One command starts the same
   pinned version on Linux, macOS, and Windows, without installing its
   Python dependencies next to the course's.

## Letting a model do the first pass: pseudo-labeling and friends

Labeling everything by hand is slow, so people often let a model label
first and correct or filter its output. The technique you may have heard of,
**using a model to label unlabeled data**, is called **pseudo-labeling**
(the model's predictions are *pseudo-labels*). When the labeling model is a
bigger, stronger model and its labels train a smaller one, it is called
**knowledge distillation** or **teacher-student training**. In annotation
tools, the same idea is sold as **auto-labeling** or **model-assisted
labeling**. The main variants:

| Technique | How it works | Example |
| --- | --- | --- |
| **Pseudo-labeling** (self-training) | Train a model on the few labeled images, predict the unlabeled ones, keep the confident predictions as labels, retrain on everything, repeat | Label 100 polyps, pseudo-label 1,000 more, retrain the U-Net |
| **Knowledge distillation** (teacher-student) | A large, accurate "teacher" model labels the data; a small, fast "student" learns from those labels | A big detector labels a video so that YOLOv8n can be trained for a phone ([Noisy Student](https://arxiv.org/abs/1911.04252) is a famous example) |
| **Zero-shot auto-labeling with foundation models** | Models trained on huge datasets label new classes from a text prompt or a click, with no training on your data | [CLIP](https://github.com/openai/CLIP) picks a class from text names; [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO), [OWLv2](https://huggingface.co/docs/transformers/model_doc/owlv2), or [YOLO-World](https://docs.ultralytics.com/models/yolo-world/) draw boxes for "zebra"; [Segment Anything (SAM 2)](https://github.com/facebookresearch/sam2) turns a click or a box into a mask; [autodistill](https://github.com/autodistill/autodistill) chains these to label a dataset and train YOLO |
| **Model-assisted labeling** (pre-annotation, human in the loop) | A model proposes labels; a human accepts, corrects, or deletes them. Much faster than labeling from scratch, and the human keeps control of quality | Label Studio's ML backend, Roboflow's Label Assist, CVAT's SAM tool |
| **Active learning** | The model picks the images it is least sure about, and humans label those first, so every human label teaches the most | Label the 50 images where the U-Net's mask is most uncertain |
| **Weak supervision** | Many cheap, noisy rules ("labeling functions") vote on each label, and a model learns how much to trust each rule | [Snorkel](https://github.com/snorkel-team/snorkel); hashtags as image labels |
| **Semi-supervised learning** | Training methods that use labeled and unlabeled data together (pseudo-labeling is one of them) | FixMatch, Mean Teacher |
| **Crowdsourcing** | Many non-expert people label each item, and their answers are combined by majority vote or agreement | Amazon Mechanical Turk; ImageNet was labeled this way |
| **Synthetic data** | Render or generate images whose labels are known by construction | Game engines for driving scenes; generated images |

**The risk of all of them:** a model's mistakes become labels, and the next
model learns them as truth (*confirmation bias*). Keep a test set labeled by
humans, filter pseudo-labels by confidence, and check a sample by hand.

## The three datasets

Small, easy samples that anyone can label in about an hour in total, with no
expert knowledge. The script keeps the original labels apart, so you can
measure how close your labels are afterwards.

| Task | Dataset | Images | Classes | What you label | Template | Source and license |
| --- | --- | --- | --- | --- | --- | --- |
| Classification | [Imagenette](https://github.com/fastai/imagenette) (160 px version, validation images) | 30 (10 per class), renamed `img_001.jpg`... so the name does not give away the class | dog, church, parachute | Pick one class per image (keys 1, 2, 3) | [`classification.xml`](label_configs/classification.xml) | fast.ai (Apache 2.0); a subset of ImageNet, for research and education |
| Semantic segmentation | [Balloon](https://github.com/matterport/Mask_RCNN/releases/tag/v2.1) (from the Mask R-CNN tutorial) | 20 (69 balloons) | balloon | Outline every balloon with a polygon, or paint it with a brush | [`segmentation_polygon.xml`](label_configs/segmentation_polygon.xml), [`segmentation_brush.xml`](label_configs/segmentation_brush.xml) | Matterport Mask R-CNN repository (MIT); photos from Flickr |
| Object detection (YOLO) | [African Wildlife](https://docs.ultralytics.com/datasets/detect/african-wildlife/) (test images) | 30 (7 or 8 per animal) | buffalo, elephant, rhino, zebra | Draw a box around every animal (keys 1 to 4) | [`detection.xml`](label_configs/detection.xml) | Ultralytics (AGPL-3.0) |

Why these: the objects are easy to recognize, balloons have clear outlines,
and the animals are big; with only 20 to 30 images per task you can finish,
then compare with a classmate.

## Step 1: download the data

From the repository root, with any Python 3.10 or newer (the script uses only
the standard library, so the course `.venv` is not required):

```bash
python DataAnnotation/download_data.py
```

It downloads about 240 MB once (to `data/annotation/downloads/`, reused on
the next run) and writes:

```
data/annotation/
├── classification/
│   ├── images/          # label these
│   └── ground_truth/    # labels.csv: the true class of each image
├── segmentation/
│   ├── images/
│   └── ground_truth/    # via_region_data.json: the original polygons (VGG Image Annotator format)
├── detection/
│   ├── images/
│   └── ground_truth/    # labels/*.txt (YOLO format) and classes.txt
└── downloads/           # the original archives
```

`--seed` picks which images (42 by default, the same images for everyone), and
`--per-class`, `--segmentation`, and `--detection` change how many. Running
the script again rewrites the three folders. **Run it before starting Label
Studio**, see [Troubleshooting](#troubleshooting).

Don't look in `ground_truth/` until you have finished labeling.

## Step 2: run Label Studio with Docker

**Install Docker** first: [Docker Desktop](https://docs.docker.com/get-started/get-docker/)
on Windows and macOS (start it before the commands below), or
[Docker Engine](https://docs.docker.com/engine/install/) on Linux. Check with
`docker --version` and `docker compose version`.

**Optional: your login in `.env`.** Add two lines to `.env` at the repository
root (never to `.envcolab`, which is tracked by git):

```bash
LABEL_STUDIO_USERNAME=you@example.com   # must look like an email address
LABEL_STUDIO_PASSWORD=choose-a-password
```

At the first start, Label Studio creates this account. Without these lines,
click **Sign up** on the first page and create an account there instead. Either
way, the account exists only in your local Label Studio, not on the internet.
`compose.yaml` passes only these two values to the container, not the other
keys in `.env`.

**Start it**, from the `DataAnnotation` folder:

```bash
cd DataAnnotation
docker compose --env-file ../.env up -d --build
```

The first start downloads the Label Studio image (about 1 GB) and takes a few
minutes; later starts take seconds. When `docker compose logs -f` shows the
server running (Ctrl+C stops following the logs, not the server), open
**http://localhost:8080** and log in.

| You want to | Command (from `DataAnnotation/`) |
| --- | --- |
| Start (or restart after a reboot) | `docker compose --env-file ../.env up -d` |
| Watch the logs | `docker compose logs -f` |
| Stop, keeping your projects | `docker compose stop` |
| Remove the container, keeping your projects | `docker compose down` |
| Delete everything, including all labels | `docker compose down -v` (export first!) |

Your projects and labels are stored in the Docker volume
`dataannotation_label-studio-data`, so they survive stopping, `down`, and
reboots. Only `down -v` (or `docker volume rm`) deletes them.

## Step 3: label in Label Studio

The official guide: [labelstud.io/guide](https://labelstud.io/guide/) (start
with [Get started](https://labelstud.io/guide/get_started) and
[Labeling](https://labelstud.io/guide/labeling)).

**1. Create a project.** Click **Create Project** and give it a name, for
example *Classification*.

**2. Set up the labeling interface.** In the **Labeling Setup** tab, choose
**Custom template**, switch to the **Code** view, and replace the XML with
the content of the matching file in [`label_configs/`](label_configs/). The
preview on the right shows the interface. The XML is the interface: `<Image>`
shows the image, and `<Choices>`, `<PolygonLabels>`, `<BrushLabels>`, or
`<RectangleLabels>` add the tools and classes. Browse the
[templates gallery](https://labelstud.io/templates) for other tasks. Click
**Save**.

**3. Import the images**, in one of two ways:

- **Upload (simplest):** in the **Data Import** tab (or the **Import** button
  of an existing project), drag the files of `data/annotation/<task>/images/`
  into the window. Label Studio keeps a copy in its volume.
- **Local files (no copy):** compose.yaml makes `data/annotation` visible to
  Label Studio as `/label-studio/files/annotation`. In the project, open
  **Settings > Cloud Storage > Add Source Storage**, choose **Local files**,
  set the absolute local path to, for example,
  `/label-studio/files/annotation/detection/images`, set the file filter to
  `.*\.jpg`, turn on **Treat every bucket object as a source file**, click
  **Add Storage**, then **Sync Storage**. See
  [Local storage](https://labelstud.io/guide/storage#Local-storage).

Each image becomes a **task**, listed in the **Data Manager** (the project's
table view).

**4. Label.** Click **Label All Tasks**. For each image:

| Task | How |
| --- | --- |
| Classification | Press 1, 2, or 3 (or click the class), then submit |
| Polygon segmentation | Press 1 (or click *balloon*), click points along the edge, click the first point to close the polygon; one polygon per balloon |
| Brush segmentation | Press 1, paint over the balloon; the eraser removes paint, and the brush size can be changed in the toolbar |
| Detection | Press 1 to 4 to pick the animal, then drag a box from one corner to the opposite one |

| Shortcut | Action |
| --- | --- |
| Ctrl+Enter (Cmd+Enter on macOS) | Submit the image and go to the next one |
| Ctrl+Z | Undo |
| Backspace or Delete | Delete the selected region (box, polygon, ...) |
| 1, 2, 3, ... | Select a class (set with `hotkey` in the XML) |
| Mouse wheel with the zoom tools | Zoom in for precise edges |

Click a finished region to select, move, or reshape it; **Update** saves
changes to an image you already submitted; **Skip** leaves an image for later.

**5. Follow the guidelines.** The header of each interface states the rule;
here is the full list:

- **Classification:** the class of the main object, even if it is small.
- **Segmentation:** every balloon, including its knot but not its string; a
  balloon partly hidden by another one gets only its visible part.
- **Detection:** a tight box around the visible part of every animal, even if
  it is small, partly hidden, or cut by the image border; one box per animal,
  never one box around a group.

## Step 4: export, and check your labels

In the Data Manager, click **Export** and pick a format; Label Studio lists
only the formats that fit the project's interface. The export downloads as a
file through your browser.

| Task | Useful exports | Then |
| --- | --- | --- |
| Classification | CSV, JSON | Compare with `ground_truth/labels.csv`: your accuracy |
| Polygon segmentation | COCO, JSON | Turn polygons into masks (for example with Pillow's `ImageDraw.polygon`) and compute the Dice with masks drawn from `via_region_data.json` |
| Brush segmentation | Brush labels to PNG, Brush labels to NumPy | One mask per image, ready for a U-Net |
| Detection | YOLO | A `classes.txt` and one `.txt` per image in the format of [YoloExample](../YoloExample/); compare your boxes with `ground_truth/labels/` by IoU |

The YOLO export numbers the classes in alphabetical order (buffalo 0,
elephant 1, rhino 2, zebra 3), which is also the order of the dataset's own
labels in `ground_truth/classes.txt`, so the class numbers can be compared
directly.

**Things to try:**

1. **Measure yourself:** your classification accuracy, your mean box IoU, and
   your mean mask Dice against the ground truth. Which task was hardest to do
   precisely?
2. **Inter-annotator agreement:** label the same images as a classmate and
   compute the Dice and IoU between your two sets of labels. How does it
   compare with the scores of the models in this repository?
3. **Time it:** how long does one box take, one polygon, one brushed mask?
   Multiply by the 1,000 images of Kvasir-SEG.
4. **Polygon or brush?** Label the same 5 balloons both ways. Which is faster,
   and which gives the higher Dice?
5. **Let a model help:** run a COCO-pretrained YOLO (as in YoloExample) on the
   detection images, import its boxes as predictions, and only correct them.
   How much faster is it? (See [Import pre-annotated data](https://labelstud.io/guide/predictions)
   and the [ML backend](https://labelstud.io/guide/ml) guides.)

## Docker cheatsheet

**Docker** runs an application in a **container**: an isolated environment
that holds the application and everything it needs (here Label Studio, its
Python packages, and its web server), so it runs the same on every computer.
An **image** is the read-only template a container is started from; a
`Dockerfile` describes how to build an image; a **volume** is storage that
outlives containers; and **Docker Compose** starts containers from a
`compose.yaml` file instead of a long command line.

| Command | What it does | Why you do it |
| --- | --- | --- |
| `docker --version`, `docker compose version` | Print the installed versions | Check that Docker is installed and running |
| `docker pull heartexlabs/label-studio:1.23.2` | Download an image from Docker Hub | Get the application without building it (`build` and `up` do this for you) |
| `docker build -t cs4337-label-studio .` | Build an image from the `Dockerfile` in the current folder, named with `-t` | Create your own image on top of an official one (here: pinned version and settings) |
| `docker images` | List the images on your computer, with their sizes | See what is downloaded and how much disk it uses |
| `docker run -d -p 8080:8080 -v label-studio-data:/label-studio/data cs4337-label-studio` | Start a container from an image: `-d` in the background, `-p host:container` publishes a port, `-v volume:path` attaches storage | Run an image without Compose (compose.yaml writes these options down for you) |
| `docker ps`, `docker ps -a` | List running containers (`-a`: also stopped ones) | See whether Label Studio is running and under which name |
| `docker logs -f cs4337-label-studio` | Show a container's output, `-f` to keep following it | Read startup messages and errors |
| `docker exec -it cs4337-label-studio sh` | Open a shell inside a running container | Look at files or run commands inside it; `exit` leaves |
| `docker stop cs4337-label-studio`, `docker start cs4337-label-studio` | Stop and restart a container, keeping its state | Free memory when you are not labeling |
| `docker rm cs4337-label-studio` | Delete a stopped container (not its volumes) | Clean up before recreating it with new settings |
| `docker rmi cs4337-label-studio:1.23.2` | Delete an image | Free disk space (about 1 GB) after the course |
| `docker volume ls`, `docker volume rm <name>` | List and delete volumes | Find where your projects live; `rm` deletes all labels in it for good |
| `docker system df` | Show the disk space used by images, containers, and volumes | Find out what fills your disk |
| `docker system prune` | Delete stopped containers, unused networks, and dangling images (not volumes unless `--volumes`) | Clean up after experimenting |
| `docker compose up -d --build` | Build the image if needed and start every service of `compose.yaml` in the background | The one command to start Label Studio |
| `docker compose --env-file ../.env up -d` | The same, reading variables such as `LABEL_STUDIO_USERNAME` from another `.env` file | Use the repository's `.env` instead of one in this folder |
| `docker compose ps` | List this project's containers and their state | Check that the service is up |
| `docker compose logs -f` | Follow the logs of this project's services | Same as `docker logs`, without typing the container name |
| `docker compose stop`, `docker compose start` | Stop and start the services, keeping containers | Pause and resume |
| `docker compose down` | Stop and remove the containers and network (volumes are kept) | Clean up, or recreate the container after editing `compose.yaml` |
| `docker compose down -v` | Also delete the volumes | Start over from zero: **deletes all projects and labels** |
| `docker compose config` | Print compose.yaml with every variable filled in | Check that the values from `.env` are picked up |

## Troubleshooting

- **`port is already allocated` (port 8080 in use).** Another program uses
  port 8080. In `compose.yaml`, change `"8080:8080"` to `"8081:8080"` and open
  http://localhost:8081 instead.
- **`data/annotation` belongs to root, or the script fails with permission
  denied.** If Label Studio was started before `download_data.py` ran, Docker
  created the missing `data/annotation` folder as root. Stop Label Studio,
  run `sudo chown -R $USER data/annotation` (Linux), then run the script.
- **`permission denied ... docker.sock` (Linux).** Your user may not run
  Docker yet: `sudo usermod -aG docker $USER`, then log out and in again.
- **`Cannot connect to the Docker daemon` (Windows, macOS).** Start Docker
  Desktop and wait until it says it is running.
- **The account from `.env` does not work.** It is created only at the first
  start, when the volume is new; changing `.env` later does not change it. To
  set a new password:
  `docker compose exec label-studio label-studio reset_password --username you@example.com --password new-password`.
- **The startup message says "Update available 1.23.0 → 1.23.2".** The
  1.23.2 image reports its package as 1.23.0; the server itself is 1.23.2
  (see `http://localhost:8080/api/version`). Ignore it.
- **No Docker?** Label Studio also installs with pip, in its own virtual
  environment so its packages do not clash with the course's:
  `python -m venv .venv-ls`, activate it, `pip install label-studio==1.23.2`,
  then `label-studio start`. Upload the images in the browser.

## Links

- Label Studio website: [labelstud.io](https://labelstud.io)
- User guide: [labelstud.io/guide](https://labelstud.io/guide/)
- Installing with Docker: [labelstud.io/guide/install#Install-with-Docker](https://labelstud.io/guide/install#Install-with-Docker)
- Labeling interface tags (`<Image>`, `<Choices>`, `<RectangleLabels>`, ...): [labelstud.io/tags](https://labelstud.io/tags/)
- Templates gallery: [labelstud.io/templates](https://labelstud.io/templates)
- Exporting: [labelstud.io/guide/export](https://labelstud.io/guide/export)
- Pre-annotations and the ML backend: [labelstud.io/guide/predictions](https://labelstud.io/guide/predictions), [labelstud.io/guide/ml](https://labelstud.io/guide/ml)
- Source code: [github.com/HumanSignal/label-studio](https://github.com/HumanSignal/label-studio)
- Docker documentation: [docs.docker.com](https://docs.docker.com)
