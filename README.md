# Titre du projet : Semantic head motion prediction in Virtual Reality videos
- Encadrant :  [Frédéric Preciosos](mailto:frederic.precioso@.fr), Laboratoire I3S 
- Co-encadrants : [Miguel Romero](mailto:miguelfromeror@gmail.com) @MiguelFRomeroR et [Lucile Sassatelli](mailto:sassatelli@i3s.unice.fr) @sassatelli
- Nom de l'étudiant : Mohamed YOUNES SI5 Parcours SD
## Résumé du contexte et le périmètre de l'étude à réaliser :
#### L’adoption massive de la réalité virtuelle représente un défi majeur : le streaming sur Internet. 
#### En effet, en raison de la proximité de l'écran à l'œil dans les visiocasques VR et de la largeur de la vue omnidirectionnelle de la scène, le débit de données est de deux ordres de grandeur pour une vidéo normale.[1]
#### Un autre problème est que ce qui est affiché sur l'appareil (le champ de vision - FoV) ne représente qu'une fraction de ce qui est téléchargé, et ce gaspillage de bande passante est le prix à payer pour l'interactivité [2].
#### Pour diminuer la quantité de données à diffuser, des travaux récents ont proposé d’envoyer en haute qualité uniquement la zone de la sphère vidéo correspondant au FoV [3-5]. Ces approches nécessitent toutefois de connaître à l’avance la position de la tête de l’utilisateur, c’est-à-dire au moment de l’envoi du contenu depuis le serveur. Il est donc crucial pour un système de streaming de vidéos 360 ° d’intégrer un prédicteur de mouvement de la tête précis, capable d’informer périodiquement, en fonction de la trajectoire passée et du contenu, que l’utilisateur sera susceptible de regarder au-delà de l’horizon.
#### Diverses méthodes abordant ce problème avec des réseaux de neurones profonds ont été proposées au cours des deux dernières années [6-10]. Dans une étude récente, nous avons constaté que toutes les méthodes existantes pour ce problème ne profitent pas des informations sur le contenu et sont surperformés par une ligne de base triviale qui ne suppose aucun mouvement.
#### Pour résoudre le problème de l'extraction d'informations précieuses du contenu, nous proposons d'intégrer des informations de dernière génération à partir de la saisie visuelle dans les modèles Deep Learning (DL) ; au lieu de saisir directement les images à 360 °, nous proposons d’utiliser des informations sémantiques telles que la position des objets dans la scène.
#### Le but de ce projet est d’aider à la création de ce modèle DL et de cette nouvelle proposition de problème. Au lieu de prédire la trajectoire réelle, nous voudrions prédire quels objets de la scène sphérique seront suivis par les utilisateurs.
### Défis identifiés :
- Se familiariser avec le problème de prédiction de mouvement de la tête pour un contenu immersif
et comprendre la solution proposée.
- Implémenter la méthode de référence de CVPR’18 [11] et la comparer à la méthode que déjà
développée.
- Comparer les deux méthodes avec d'autres techniques (architectures utilisant la position de tête et
les cartes de saillance).
- Transformer les traces numériques en traces sémantiques :
- Comprendre le code pour détecter et suivre tous les objets de la scène 3D.
- Définir un protocole pour déterminer si un objet est suivi par le spectateur ou non.
- Modéliser le problème en utilisant un RNN pour prédire les futures positions de focalisation de la
tête.
- Tests, analyses et améliorations : Comparer les résultats avec d'autres techniques (architectures
utilisant la position de tête et des cartes de saillance sémantique).
- Étudier le tout nouveau problème de l'extraction de saillance de l’audio, en plus de la modalité
visuelle.

## Références :
* [1] International Data Corporation., “Demand for augmented reality/virtualreality headsets expected
to rebound in 2018,” Industry report, 2018.
* [2] Corbillon, X., Simon, G., Devlic, A., & Chakareski, J. (2017, May). Viewport-adaptive navigable
360-degree video delivery. In 2017 IEEE international conference on communications (ICC) (pp. 1-7).
IEEE.
* [3] Qian, F., Ji, L., Han, B., & Gopalakrishnan, V. (2016, October). Optimizing 360 video delivery over
cellular networks. In Proceedings of the 5th Workshop on All Things Cellular: Operations,
Applications and Challenges (pp. 1-6). ACM.
* [4] Xiao, M., Zhou, C., Swaminathan, V., Liu, Y., & Chen, S. (2018, April). Bas-360: Exploring spatial
and temporal adaptability in 360-degree videos over http/2. In IEEE INFOCOM 2018-IEEE Conference
on Computer Communications (pp. 953-961). IEEE.
* [5] Hristova, H., Corbillon, X., Simon, G., Swaminathan, V., & Devlic, A. (2018, August).
Heterogeneous Spatial Quality for Omnidirectional Video. In 2018 IEEE 20th International Workshop
on Multimedia Signal Processing (MMSP) (pp. 1-6). IEEE.
* [6] Xu, M., Song, Y., Wang, J., Qiao, M., Huo, L., & Wang, Z. (2018). Predicting head movement in
panoramic video: A deep reinforcement learning approach. IEEE transactions on pattern analysis and
machine intelligence.
* [7] Xu, Y., Dong, Y., Wu, J., Sun, Z., Shi, Z., Yu, J., & Gao, S. (2018). Gaze prediction in dynamic 360
immersive videos. In proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition (pp. 5333-5342).
* [8] Nguyen, A., Yan, Z., & Nahrstedt, K. (2018, October). Your attention is unique: Detecting 360-
degree video saliency in head-mounted display for head movement prediction. In 2018 ACM
Multimedia Conference on Multimedia Conference (pp. 1190-1198). ACM.
* [9] Li, Y., Xu, Y., Xie, S., Ma, L., & Sun, J. (2018, October). Two-Layer FoV Prediction Model for
Viewport Dependent Streaming of 360-Degree Videos. In International Conference on
Communications and Networking in China (pp. 501-509). Springer, Cham.
* [10] Fan, C. L., Lee, J., Lo, W. C., Huang, C. Y., Chen, K. T., & Hsu, C. H. (2017, June). Fixation prediction
for 360 video streaming in head-mounted virtual reality. In Proceedings of the 27th Workshop on
Network and Operating Systems Support for Digital Audio and Video (pp. 67-72). ACM.
* [11] Yanyu Xu, Yanbing Dong, Junru Wu, Zhengzhong Sun, Zhiru Shi, Jingyi Yu, and Shenghua Gao
ShanghaiTech University. Gaze Prediction in Dynamic 360◦ Immersive Videos.
http://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_Gaze_Prediction_in_CVPR_2018_paper.pdf
