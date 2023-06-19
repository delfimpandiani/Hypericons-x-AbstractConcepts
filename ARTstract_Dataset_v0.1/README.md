# ARTstract Dataset

The ARTstract dataset is a collection of images of artworks, primarily focused on abstract concepts. The images were sourced from a variety of other art datasets, including ArtPedia, ARTEMIS, ADVISE and the Tate Collection. 

## Abstract Concepts and Evoked Clusters

The ARTstract dataset uses evoked clusters as a way to label the abstract concepts present in each image. Evoked clusters are groups of abstract concepts that often co-occur together in a given context. The idea of abstract concepts or symbols functioning in clusters of evocation was first introduced in [Hussain et al 2017], and has been used the scarce research there is on abstract concept detection with computer vision, such as [Ye et al 2018] and [Kalanat and Kovashka 2022]. It's important to note that currently the only available performances for the task of abstract concept detection use these clusters, that's why we have decided to use them to create this dataset. The original clusters were created by [Hussain et al 2017] by analyzing the co-occurrence of abstract concepts in advertisement images. The goal is to group similar abstract concepts together, in order to make it easier to detect and understand them in new images.

We have reused those clusters and selected only 12 of them to begin populating the dataset. These 12 were selected by cross-referencing the cluster names (comfort, freedom, etc.) with cognitive science data about abstract concepts. We selecting the 12 (out of 55) clusters that were explicitly present in one of the latest cognitive science datasets [Harpaintner et al 2020], namely:

- danger: danger, peril, risk
- death: death, lethal, suicide, funeral
- adventure: adventure, skiing, sport
- power: force, power, powerful
- safety: safety, saftey, security
- excitement: excitement, flavors
- fitness: exercise, fitness, running
- protection: condom, penis, protection
- freedom: america, freedom, liberty
- comfort: comfort, cozy, soft, softness
- desire: desire, want
- hunger: desperation, hunger

Each image in the ARTstract dataset is assigned one or more evoked clusters based on the abstract concepts present in the image. The evoked clusters are stored in the ARTstract_img_dict.json file, in the evoked_clusters field for each image. Each evoked cluster is represented by a number, and the corresponding cluster name, words, evocation strength and evocation evidence can be found in the same JSON file.

## Data Description

The dataset contains X images, with Y abstract concepts as tags. Each image is labeled with one or more tags that describe the abstract concepts present in the artwork. The images are in JPG format and have a resolution of ZxZ pixels. The labels for each image are stored in a JSON file called ARTstract_img_dict.json. The JSON file contains a dictionary where each key is a string representing the image name and the value is a dictionary containing the following information:

- source_dataset: The source dataset from which the image was obtained.
- source_id: The id of the image in the source dataset.
- evoked_clusters: A dictionary containing information about the abstract concepts evoked by the image. Each key is a cluster number, and the value is a dictionary containing the following information:
- cluster_name: The name of the cluster.
- cluster_words: A list of words that are associated with the cluster.
- evocation_strength: A number indicating the strength of evocation of the cluster by the image.
- evocation_evidence: A list of sentences that provide evidence for the evocation of the cluster by the image.

```
{
   "artpedia_3629":{
      "source_dataset":"artpedia",
      "source_id":"3629",
      "evoked_clusters":{
         "4":{
            "cluster_name":"danger",
            "cluster_words":[
               "danger",
               "peril",
               "risk"
            ],
            "evocation_strength":1,
            "evocation_evidence":[
               "She straddles the seas, uniting the continents, or hovers over the harbor to protect ships, cargo and crew as they embark on the perilous Atlantic crossing."
            ]
         }
      }
   },
   "artpedia_4963":{
      "source_dataset":"artpedia",
      "source_id":"4963",
      "evoked_clusters":{
         "4":{
            "cluster_name":"danger",
            "cluster_words":[
               "danger",
               "peril",
               "risk"
            ],
            "evocation_strength":1,
            "evocation_evidence":[
               "Like the world these apostles knew, the basket of food teeters perilously over the edge."
            ]
         }
      }
   },
   ...
}

```

## Data Usage
The ARTstract dataset can be used for various computer vision tasks, such as image classification, object detection, and semantic segmentation. The abstract concept tags can also be used for training models for image captioning and other natural language processing tasks.

## Dataset Creation
The dataset was created by Delfina Sol Martinez Pandiani in 2023, by  mining a variety of other art datasets, including ArtPedia, ARTEMIS, ADVISE and the Tate Collection. The images were selected based on their relevance to abstract concepts and their high resolution. The tags for each image were manually assigned by experts in the field of art and abstract concepts.

## Licensing
The ARTstract dataset is NOT yet available for research and educational use.

## Contact
For any questions or issues related to the dataset, please contact Delfina Sol Martinez Pandiani at delfinasol.martinez2@unibo.it.



## Data Statistics


| ****             | ****             | ****               | ****                   | **Artpedia**               | ****                       | ****                              | ****                              | **ADVISE**               | ****                     | ****                            | ****                            | **ARTEMIS**               | ****                      | ****                             | ****                             | **Tate**               | ****                   | ****                          | ****                          |
|------------------|------------------|--------------------|------------------------|----------------------------|----------------------------|-----------------------------------|-----------------------------------|--------------------------|--------------------------|---------------------------------|---------------------------------|---------------------------|---------------------------|----------------------------------|----------------------------------|------------------------|------------------------|-------------------------------|-------------------------------|
| ****             |                  |                    |                        | Artpedia Total             |                            | Artpedia Unique                   |                                   | ADVISE Total             |                          | ADVISE Unique                   |                                 | ARTEMIS Total             |                           | ARTEMIS Unique                   |                                  | Tate Total             |                        | Tate Unique                   |                               |
| **cluster_name** | Total_count      | Total_UNIQUE_count | Percent that is UNIQUE | artpedia_count_evoker_imgs | artpedia_evocation_average | artpedia_count_UNIQUE_evoker_imgs | artpedia_UNIQUE_average_evocation | advise_count_evoker_imgs | advise_evocation_average | advise_count_UNIQUE_evoker_imgs | advise_UNIQUE_average_evocation | artemis_count_evoker_imgs | artemis_evocation_average | artemis_count_UNIQUE_evoker_imgs | artemis_UNIQUE_average_evocation | tate_count_evoker_imgs | tate_evocation_average | tate_count_UNIQUE_evoker_imgs | tate_UNIQUE_average_evocation |
| **danger**       | 2463             | 1440               | 58.4652862362972       | 15                         | 1.0                        | 10                                | 1.0                               | 1072                     | 1.398320895522390        | 531                             | 1.3822975517890800              | 1336                      | 1.06437125748503          | 879                              | 1.0295790671217300               | 40                     | 1.0                    | 20                            | 1.0                           |
| **death**        | 3837             | 2824               | 73.599166015116        | 90                         | 1.1222222222222200         | 84                                | 1.119047619047620                 | 658                      | 1.6838905775076000       | 309                             | 1.6019417475728200              | 2278                      | 1.1795434591747100        | 1732                             | 1.153579676674370                | 811                    | 1.0                    | 699                           | 1.0                           |
| **adventure**    | 11800            | 10885              | 92.2457627118644       | 11                         | 1.0909090909090900         | 9                                 | 1.1111111111111100                | 942                      | 1.3142250530785600       | 624                             | 1.310897435897440               | 1177                      | 1.076465590484280         | 728                              | 1.0425824175824200               | 9670                   | 1.0                    | 9524                          | 1.0                           |
| **power**        | 3791             | 2744               | 72.3819572672118       | 66                         | 1.0454545454545500         | 59                                | 1.0508474576271200                | 582                      | 1.2577319587628900       | 426                             | 1.2793427230046900              | 3096                      | 1.0704134366925100        | 2218                             | 1.0554553651938700               | 47                     | 1.0                    | 41                            | 1.0                           |
| **safety**       | 707              | 358                | 50.6364922206506       | 4                          | 1.0                        | 2                                 | 1.0                               | 384                      | 1.2265625                | 178                             | 1.2134831460674200              | 312                       | 1.0288461538461500        | 172                              | 1.005813953488370                | 7                      | 1.0                    | 6                             | 1.0                           |
| **excitement**   | 2046             | 1375               | 67.2043010752688       | 1                          | 1.0                        | 1                                 | 1.0                               | 308                      | 1.1233766233766200       | 196                             | 1.1224489795918400              | 1736                      | 1.0339861751152100        | 1178                             | 1.0271646859083200               | 1                      | 1.0                    | 0                             | 1.0                           |
| **fitness**      | 1230             | 835                | 67.8861788617886       | 15                         | 1.0                        | 12                                | 1.0                               | 236                      | 1.2966101694915300       | 168                             | 1.3392857142857100              | 858                       | 1.083916083916080         | 576                              | 1.0607638888888900               | 121                    | 1.0                    | 79                            | 1.0                           |
| **protection**   | 537              | 334                | 62.1973929236499       | 6                          | 1.0                        | 4                                 | 1.0                               | 259                      | 1.2702702702702700       | 146                             | 1.2397260273972600              | 256                       | 1.109375                  | 169                              | 1.118343195266270                | 16                     | 1.0                    | 15                            | 1.0                           |
| **freedom**      | 730              | 510                | 69.8630136986301       | 6                          | 1.1666666666666700         | 6                                 | 1.1666666666666700                | 288                      | 1.1979166666666700       | 201                             | 1.2189054726368200              | 393                       | 1.0330788804071200        | 268                              | 1.0223880597014900               | 43                     | 1.0                    | 35                            | 1.0                           |
| **comfort**      | 7810             | 6227               | 79.7311139564661       | 34                         | 1.0                        | 28                                | 1.0                               | 273                      | 1.2747252747252700       | 213                             | 1.3098591549295800              | 7423                      | 1.1158561228613800        | 5923                             | 1.0913388485564700               | 80                     | 1.0                    | 63                            | 1.0                           |
| **desire**       | 7275             | 5482               | 75.3539518900344       | 12                         | 1.0                        | 9                                 | 1.0                               | 268                      | 1.0895522388059700       | 222                             | 1.0945945945945900              | 6950                      | 1.081726618705040         | 5217                             | 1.048111941728960                | 45                     | 1.0                    | 34                            | 1.0                           |
| **hunger**       | 426              | 311                | 73.0046948356808       | 1                          | 1.0                        | 0                                 | 1.0                               | 241                      | 1.174273858921160        | 196                             | 1.1938775510204100              | 183                       | 1.010928961748630         | 114                              | 1.0                              | 1                      | 1.0                    | 1                             | 1.0                           |
| **AVERAGE**      | 3554.33333333333 | 2777.08333333333   | 70.2141093077216       | 21.75                      | 1.03543771043771           | 18.6666666666667                  | 1.03730607120438                  | 459.25                   | 1.27562134059408         | 284.166666666667                | 1.27555500823231                | 2166.5                    | 1.07404231170301          | 1597.83333333333                 | 1.05459342500926                 | 906.833333333333       | 1.0                    | 876.416666666667              | 1.0                           |
