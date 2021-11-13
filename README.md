# BEGAN-CS_with_Triplet_Loss
- dataset: asian_celeb
- output images: (3, 64, 64)
- data parallelism
- combine the triplet loss with the constraint loss
- d_total_loss = d_loss + alpha * latent_constraint + beta * triplet_loss
- beta = 0.15
- replace triplet loss with cosine similarity