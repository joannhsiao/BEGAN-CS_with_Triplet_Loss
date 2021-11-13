# BEGAN-CS with Triplet Loss
- dataset: asian_celeb
- output images: (3, 128, 128)
- data parallelism
- combine the triplet loss with the constraint loss
- d_total_loss = d_loss + alpha * latent_constraint + beta * triplet_loss
- alpha = 0.1
- beta = 0.25
- Margin = 0.5
- batch_size = 32
- LMS cannot be utilized
- output: euclidean distance & cosine distance


# Experiments
1. alpha = 0.1, beta = 0.2/0.15 -> [100] cosine distance is not improve much (lab meeting ppt)
2. alpha = 0.1, beta = 0.2, charlie = 0.05 -> [7] seems mode collapse (lab meeting ppt)
3. alpha = 0.1, beta = 0.2, charlie = 0.01 -> [7] generator crashed; mode collapse (3090)
4. alpha = 0.1, beta = 0.5 -> [4] mode collapse (cv307)
5. alpha = 0.1, beta = 0.5 + bn on G -> [24] converge to some modes (3090)
6. alpha = 0.1, beta = 0.5 + bn on G/D -> [5] collapse to some modes (3090)
7. alpha = 0.1, beta = 0.25 + bn on G/D -> [6] collapse to some modes (cv307)
8. alpha = 0.1, beta = 0.25 + bn on G -> [3] mode collapse (cv307; only an image)
9. alpha = 0.1, beta = 0.25, charlie = 0.005 -> [4] crashed (3090)
10. identification: resnet, alpha = 0.2, beta = 0.3, charlie = 0.1 -> [11] cosine distance not improve much (3090)
11. triplet loss, alpha = 0.1, beta = 0.2 -> mode collapse when epoch = 16(3090)
12. identification: resnet, triplet loss, sn on G/D, alpha = 0.1, beta = 0.2, charlie = 0.05 -> D crashed (cv307)
13. identification: resnet, tripler loss, sn on G, alpha = 0.1, beta = 0.2, charlie = 0.05 -> better(cv307)
14. triplet loss, sn on G, alpha = 0.1, beta = 0.2
