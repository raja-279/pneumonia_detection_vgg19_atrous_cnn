**# Automated Pneumonia Detection with a Hybrid VGG19 and Atrous CNN Approach**



An educational deep learning project that classifies chest X-ray images as **\*\*NORMAL\*\*** or **\*\*PNEUMONIA\*\*** using a pretrained VGG19 network with extra atrous (dilated) convolution layers on top, built with TensorFlow/Keras.



\> ⚠️ **\*\*This is an academic project, not a medical tool.\*\*** It must not be used to diagnose, screen or treat anyone. See the [Disclaimer]\(#disclaimer).



**\*\*Quick links:\*\*** [About]\(#-about-the-project) · [Architecture]\(#-model-architecture) · [Dataset]\(#-dataset) · [Setup]\(#-installation-and-setup) · [Training]\(#-training-the-model) · [Predictions]\(#-using-the-trained-model) · [Model download]\(#-model-download) · [Results]\(#-results-and-evaluation) · [Limitations]\(#-limitations)



**---**



**## 📖 About the project**



Chest X-rays are one of the most common ways to check for pneumonia, but reading them takes training and time, and trained radiologists are not available everywhere. This project was my attempt to see how well a deep learning model could separate pneumonia X-rays from normal ones as a binary classification problem.



The idea is to take a VGG19 network that was already trained on ImageNet and reuse its feature extraction layers. On top of that I added a small block of convolution layers, including atrous (dilated) convolutions, which look at a wider area of the image without adding more parameters per layer. The final layers reduce everything to a single number between 0 and 1. A value of 0.5 or higher is read as PNEUMONIA, anything lower as NORMAL.



The repository contains scripts for splitting the data, training and evaluating the model, along with the CSV of test predictions. The trained model will be provided separately through a GitHub Release once that release is published (see [Model download]\(#-model-download)).



**## Features**



Only what is actually in this repository:



\- A script that splits a folder of images into train, validation and test sets (70/15/15).

\- A training script with data augmentation, a frozen VGG19 base, atrous convolution layers and early stopping.

\- Evaluation on the test set, with a classification report and confusion matrix printed in the terminal and per-image predictions saved to \`pneumonia_results.csv\`.

\- A prediction script that classifies every image in the \`demo_images/\` folder and prints the label, a confidence percentage and the raw model probability.

\- A small demo script that loads the model, prints its architecture and classifies one image.



There is no web app or GUI. Everything runs from the command line.



**## Technologies used**



\| Tool | Used for |

\|---|---|

\| Python 3.10 | Language (version listed in \`environment.txt\`) |

\| TensorFlow 2.10.0 / Keras 2.10.0 | Model building, training and prediction |

\| VGG19 (ImageNet weights) | Pretrained feature extractor |

\| NumPy 1.23.5 | Image array handling |

\| pandas | Saving the test predictions to CSV |

\| scikit-learn | Classification report and confusion matrix |

\| Pillow, SciPy | Needed by Keras for loading images and for augmentation (not imported directly) |



Versions for TensorFlow, Keras and NumPy come from \`environment.txt\`. I did not record exact versions for the other packages, so \`requirements.txt\` leaves them unpinned.



**## 🧠 Model architecture**



The model is defined in \`train_vgg19_atrous.py\`. Input images are 224 × 224 RGB.



\| Stage | Layers | Notes |

\|---|---|---|

\| Base | VGG19, ImageNet weights, no top | **\*\*All VGG19 layers are frozen\*\***, so they are not updated during training |

\| Extra conv block | Conv2D (128, 3×3, ReLU) → BatchNorm → MaxPooling → Dropout (0.4) | |

\| Atrous block | Conv2D (64, 3×3, ReLU) with dilation rate **\*\*1\*\***, then **\*\*2\*\***, then **\*\*4\*\*** | Larger dilation rates widen the area each filter covers |

\| Pooling | BatchNorm → Global Average Pooling | |

\| Classifier | Dense 256 (ReLU) → Dropout 0.5 → Dense 128 (ReLU) → Dropout 0.5 | |

\| Output | Dense 1, sigmoid | Probability of PNEUMONIA |



In plain terms: VGG19 finds general visual patterns, the extra block and the dilated convolutions look at those patterns at a few different scales, and the dense layers turn them into a yes/no answer. Batch normalization and dropout are there to reduce overfitting.



The configuration stored inside the saved model file \`best_vgg19_atrous.h5\` matches this structure.



**## 📂 Dataset**



The images come from Kaggle. The repository does **\*\*not\*\*** include any dataset images.



\- **\*\*Name:\*\*** COVID-19 Radiography Database

\- **\*\*Link:\*\*** https\://www\.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

\- **\*\*Created by:\*\*** a team of researchers from Qatar University and the University of Dhaka, with collaborators from Pakistan and Malaysia, together with medical doctors. See [References]\(#-acknowledgements-and-references).



**\*\*How it is used in this project.\*\*** The Kaggle dataset is organised into several categories (COVID-19, Normal, Lung Opacity and Viral Pneumonia). This project does **\*\*not\*\*** do COVID-19 detection or four-class classification. The code works with exactly two class folders:



\| Folder name | Class | Label used by the model |

\|---|---|---|

\| \`NORMAL\` | Normal chest X-rays | 0 |

\| \`pneumonia\` | Pneumonia chest X-rays | 1 |



Keras reads class folders in alphabetical order, so \`NORMAL\` becomes 0 and \`pneumonia\` becomes 1. The training script prints this mapping when it starts.



**\*\*What I did not record.\*\*** I did not keep a record of exactly which Kaggle subfolders or how many images went into \`NORMAL/\` and \`pneumonia/\`, or how the pneumonia images were collected, so I am not going to guess. The only size information available comes from the test set: 689 images (576 NORMAL, 113 pneumonia). If you rebuild the folders yourself, your own choice of source images will affect the results.



**\*\*Terms of use.\*\*** Please check the license and usage terms on the Kaggle page and follow them before using or redistributing the dataset.



**## 🗂️ Project structure**



\`\`\`

pneumonia_detection_vgg19_atrous_cnn/

├── README.md

├── requirements.txt

├── environment.txt             # Python / library versions I used

├── .gitignore

├── split_dataset.py            # splits NORMAL/ and pneumonia/ into train, val, test

├── train_vgg19_atrous.py       # builds, trains and evaluates the model

├── predict_xray.py             # classifies every image in demo_images/

├── load_model_demo.py          # prints the model summary, classifies one image

├── pneumonia_results.csv       # test predictions saved by the training script

├── demo_images/                # put your own X-ray images here

│   └── .gitkeep

└── best_vgg19_atrous.h5        # downloaded model file (not committed; see Model download)

\`\`\`



When you prepare the dataset locally you will also create \`NORMAL/\`, \`pneumonia/\`, \`train/\`, \`val/\` and \`test/\` folders. Git is set up to ignore these.



**## ⚙️ Installation and setup**



You will need Python 3.10 (the version listed in \`environment.txt\`) and Git. TensorFlow 2.10.0 does not support newer Python versions such as 3.11 or later.



**\*\*1. Clone the repository\*\***



\`\`\`bash

git clone https\://github.com/YOUR-USERNAME/pneumonia_detection_vgg19_atrous_cnn.git

cd pneumonia_detection_vgg19_atrous_cnn

\`\`\`



**\*\*2. Create and activate a virtual environment (Windows)\*\***



\`\`\`bash

py -3.10 -m venv venv

venv\Scripts\activate

\`\`\`



If Python 3.10 is already your default, \`python -m venv venv\` works too. If PowerShell complains that scripts are disabled, use Command Prompt instead. When the environment is active you will see \`(venv)\` at the start of the line.



**\*\*3. Install the requirements\*\***



\`\`\`bash

pip install -r requirements.txt

\`\`\`



**\*\*4. Download the trained model\*\*** (needed for predictions, see [Model download]\(#-model-download)).



\`environment.txt\` lists Windows and an Intel i5-11400F CPU as the setup. I have not tested other setups, so newer TensorFlow or Keras versions may not load the \`.h5\` file.



**## Dataset preparation**



You only need this section if you want to train the model yourself. For predictions alone you can skip it.



1\. Download the dataset from the [Kaggle page]\(https\://www\.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) (a free Kaggle account is required). The page describes the dataset as chest X-ray images plus lung masks, so make sure only the X-ray images go into your class folders, not the masks.

2\. Choose a project folder and create two folders inside it, named exactly \`NORMAL\` and \`pneumonia\` (capital letters for NORMAL, lowercase for pneumonia):



   \`\`\`

   your_project_folder/

   ├── NORMAL/        # normal X-ray images (.jpg, .jpeg or .png)

   └── pneumonia/     # pneumonia X-ray images (.jpg, .jpeg or .png)

   \`\`\`



   Which Kaggle images you put in each folder is up to you. My original selection was not recorded (see [Dataset]\(#-dataset)).



3\. Open \`split_dataset.py\` and \`train_vgg19_atrous.py\` and change this line in **\*\*both\*\*** files to your folder:



   \`\`\`python

   BASE_DIR = r"C:**\U**sers\Admin\Download&#x73;**\m**inorproj"

   \`\`\`



   For example: \`BASE_DIR = r"C:\Users\YourName\Documents\pneumonia_data"\`



4\. Run the split once:



   \`\`\`bash

   python split_dataset.py

   \`\`\`



   This **\*\*copies\*\*** (it does not move) images into \`train/\`, \`val/\` and \`test/\` inside \`BASE_DIR\`, with a 70% / 15% / 15% split and a fixed random seed of 42. Your original \`NORMAL/\` and \`pneumonia/\` folders stay as they were. If you run it again after changing the source images, delete the old \`train/\`, \`val/\` and \`test/\` folders first, otherwise old files will stay mixed in.



After the split, you should have this:



\`\`\`

your_project_folder/

├── NORMAL/

├── pneumonia/

├── train/

│   ├── NORMAL/

│   └── pneumonia/

├── val/

│   ├── NORMAL/

│   └── pneumonia/

└── test/

    ├── NORMAL/

    └── pneumonia/

\`\`\`



**## 🏋️ Training the model**



\> ⚠️ **\*\*Warning: training overwrites existing files.\*\*** \`train_vgg19_atrous.py\` saves the model as \`best_vgg19_atrous.h5\` and the results as \`pneumonia_results.csv\` in the folder you run it from. If you run it in this repository's folder, it will replace the original model and CSV. To avoid this, run it from a different copy of the project, or rename or move the original two files first. (The comment at the top of the script also marks the original model as locked.)



With the dataset prepared and \`BASE_DIR\` updated:



\`\`\`bash

python train_vgg19_atrous.py

\`\`\`



What the script does:



\| Setting | Value |

\|---|---|

\| Image size | 224 × 224 |

\| Batch size | 8 |

\| Epochs | Up to 50 (early stopping can end it sooner) |

\| Optimizer | Adam, learning rate 1e-4 |

\| Loss | Binary cross-entropy |

\| Augmentation (training only) | Rotation up to 15°, zoom 0.1, width and height shift 0.1, horizontal flip |

\| Preprocessing | Pixel values divided by 255 (training, validation and test) |

\| Early stopping | Monitors validation loss, patience 7, restores the best weights |

\| Learning rate reduction | Monitors validation loss, factor 0.3, patience 4, minimum 1e-6 |

\| Checkpoint | Saves \`best_vgg19_atrous.h5\` whenever validation accuracy improves |

\| Class weights | Not used |



When training finishes, the script evaluates on the test folder, prints a classification report and confusion matrix, and writes \`pneumonia_results.csv\` with the true label, predicted probability and predicted label for each test image. I did not record how long training took.



**## 🔍 Using the trained model**



Both scripts load \`best_vgg19_atrous.h5\` by its filename, so **\*\*run them from the repository's main folder\*\*** with the model file sitting there.



**### Classify a folder of images: \`predict_xray.py\`**



1\. Put your chest X-ray images (\`.jpg\`, \`.jpeg\` or \`.png\`) in the \`demo_images/\` folder. The repository does not ship any images, so you need to supply your own. The script stops with an error if \`demo_images/\` does not exist.

2\. Run:



   \`\`\`bash

   python predict_xray.py

   \`\`\`



For each image it prints something like this (the values here are placeholders):



\`\`\`

Image: example.jpg

Prediction: PNEUMONIA

Confidence: xx.xx%

Raw model probability: 0.xxxx

\----------------------------------------

\`\`\`



The raw probability is the model's output for the PNEUMONIA class. If it is 0.5 or higher the label is PNEUMONIA. For NORMAL results the confidence shown is 1 minus that probability. This is just the model's own output score, not a measure of clinical certainty.



**### Inspect the model and classify one image: \`load_model_demo.py\`**



This script prints the model summary and then predicts on a file called \`sample_xray.jpg\` in the main folder. That image is not included, so either add your own image with that name or change this line in the script to point to your image:



\`\`\`python

img_path = "sample_xray.jpg"

\`\`\`



It prints only the label (PNEUMONIA or NORMAL), without a confidence value.



**## 📥 Model download**



The trained model is about 87 MB (87,615 KB), which is too large for GitHub's normal file upload, so it will be made available through a **\*\*GitHub Release\*\*** rather than committed to the repository. The release still needs to be created.



1\. Open the [Releases page]\(https\://github.com/YOUR-USERNAME/pneumonia_detection_vgg19_atrous_cnn/releases).

2\. Under the latest release, download \`best_vgg19_atrous.h5\` from the "Assets" list.

3\. Put the file in the **\*\*main folder of the repository\*\***, next to \`predict_xray.py\`.



Without this file, the prediction scripts will fail with a "file not found" error.



**## 📊 Results and evaluation**



Two different figures exist for this project's test accuracy, and they come from different records.



\| Figure | Where it comes from |

\|---|---|

\| **\*\*92.4%\*\*** test accuracy | Recorded in my original project report |

\| **\*\*97.53%\*\*** accuracy (672 of 689 images) | Calculated by me from the saved \`pneumonia_results.csv\` |



These are **\*\*not\*\*** one verified experiment and **\*\*not\*\*** a measured performance range. I have not re-run the model for this repository and I am not claiming either figure has been independently verified. The two also describe different setups: the report describes a model with a Flatten layer and a Dense 512 layer trained for 10 epochs, while the code and saved model in this repository use the atrous layers, Global Average Pooling and up to 50 epochs. A comment at the top of \`train_vgg19_atrous.py\` also mentions roughly 97.7% validation and 98% test accuracy, but no training log is included to check that. The report additionally mentions a Google Colab upload interface, which is not part of this repository.



The detailed numbers below are calculated from \`pneumonia_results.csv\` only, using the 0.5 threshold. The CSV has 689 test predictions.



**\*\*Confusion matrix\*\*** (rows are the true class, columns are the predicted class)



\|  | Predicted NORMAL | Predicted PNEUMONIA |

\|---|---|---|

\| **\*\*Actual NORMAL\*\*** | 570 | 6 |

\| **\*\*Actual PNEUMONIA\*\*** | 11 | 102 |



**\*\*Per-class metrics\*\***



\| Class | Precision | Recall | F1-score | Images |

\|---|---|---|---|---|

\| NORMAL | 0.981 | 0.990 | 0.985 | 576 |

\| PNEUMONIA | 0.944 | 0.903 | 0.923 | 113 |

\| Macro average | 0.963 | 0.946 | 0.954 | 689 |

\| Weighted average | 0.975 | 0.975 | 0.975 | 689 |



Accuracy from the CSV is 672 / 689 = 97.53%.



Some context for these numbers: 83.6% of the test images are NORMAL, so a model that always answered NORMAL would already score about 83.6% accuracy. For pneumonia images, the model missed 11 of 113 (recall 0.903), which is lower than its performance on normal images. These are the numbers to look at, not the overall accuracy alone.



**## ⚠️ Limitations**



\- **\*\*Class imbalance.\*\*** The test set is mostly NORMAL images (576 vs 113), and the training script does not use class weights. Overall accuracy therefore looks better than the pneumonia-specific results.

\- **\*\*Possible patient-level overlap.\*\*** \`split_dataset.py\` splits individual images at random. If the source data contains several images from the same patient, they could land in both the training and test sets, which would make the results look better than they should. I have not checked whether this applies.

\- **\*\*The CSV may not match the saved model exactly.\*\*** The checkpoint is saved at the best validation accuracy, while early stopping restores the weights with the best validation loss, and the CSV is produced from the model in memory at the end of training. These can be different training epochs. I have not verified whether they match.

\- **\*\*Dataset selection is undocumented.\*\*** I did not record how the NORMAL and pneumonia folders were built from the Kaggle data, so the setup cannot be reproduced exactly.

\- **\*\*Single dataset.\*\*** Nothing here shows how the model behaves on X-rays from other hospitals, devices or patient groups.

\- **\*\*User-provided images.\*\*** The prediction scripts need your own images, and the repository does not include any.

\- **\*\*Environment.\*\*** Only the setup in \`environment.txt\` was used. Other TensorFlow or Keras versions are untested.

\- **\*\*Not for clinical use.\*\*** See below.



**## Disclaimer**



This is an experimental, educational deep learning project. It is **\*\*not a medical device\*\*** and has not been clinically validated. It must not be used as a substitute for professional medical diagnosis, advice or treatment. Any real concern about a chest X-ray should be reviewed by a qualified medical professional.



**## 🙏 Acknowledgements and references**



Thanks to the creators of the COVID-19 Radiography Database for making the images publicly available, and to the Kaggle community for hosting it:



\- Kaggle dataset: https\://www\.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database



The dataset creators list the following papers for citation:



1\. M. E. H. Chowdhury, T. Rahman, A. Khandakar, et al., "Can AI help in screening Viral and COVID-19 pneumonia?", *\*IEEE Access\**, vol. 8, pp. 132665 to 132676, 2020.

2\. T. Rahman, A. Khandakar, Y. Qiblawey, et al., "Exploring the Effect of Image Enhancement Techniques on COVID-19 Detection using Chest X-ray Images", arXiv:2012.02238, 2020.



VGG19 was originally introduced by Simonyan and Zisserman, and the ImageNet weights used here are the ones provided with Keras.
