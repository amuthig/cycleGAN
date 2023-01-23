Ce code est un script d'entraînement pour un modèle de génération d'images appelé CycleGAN. Il utilise PyTorch et torchvision pour gérer les images et les réseaux de neurones, ainsi que tqdm pour afficher une barre de progression pendant l'entraînement.

Le modèle CycleGAN est formé de deux discriminateurs (disc_A et disc_B) et deux générateurs (gen_A2B et gen_B2A). Les discriminateurs sont utilisés pour détecter si une image est réelle ou générée, tandis que les générateurs sont utilisés pour générer une image dans un domaine (par exemple, A) à partir d'une image dans un autre domaine (par exemple, B). Ici les domaines seront les différents types de scanner utilsés pour réaliser les IRM.

Le script définit une fonction d'entraînement appelée train_fn, qui prend en entrée les modèles, un dataloader, des optimiseurs et des fonctions de coût (L1, mse) pour entraîner les modèles. Il utilise également torch.cuda.amp.autocast() pour activer l'accélération matérielle pour les calculs sur la carte graphique.

La fonction train_fn utilise un boucle for pour parcourir les images dans le chargeur de données et entraîne d'abord les discriminateurs en utilisant la perte MSE pour distinguer les images réelles et les images générées. Ensuite, il entraîne les générateurs en utilisant la perte MSE pour maximiser la probabilité que les images générées soient considérées comme réelles par les discriminateurs. Il utilise également une perte L1 pour minimiser la différence entre les images d'entrée et les images reconstruites pour un cycle d'entraînement.

Les résultats de l'entraînement sont stockés dans une liste appelée gen_loss_list pour suivre l'évolution de la perte du générateur au fil du temps. Cette liste est ensuite affichée sous forme d'un graphique toutes les 50 itérations.

FORMAT DES DONNEES A RESPECTER:
Il est nécessaire de créer 2 dossiers vides à la racine du projet nommés: "saved_images" et "saved_model" afin que le projet enregistre les données au bon endroit. 
De plus, les données doivent être mises dans un dossier "data" situé à la racine du projet. Les chemins d'accès aux données est à adapter au niveau de la création du dataset dans le fichier "train.py"

La partie entrainement peut être lancée en utilisant le fichier cycleGAN.ipynb
Pour effectuer un test en inférence, il est nécessaire d'utiliser le fichier inference.py
