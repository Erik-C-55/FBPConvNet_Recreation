# FBPConvNet_Recreation
This is a Pytorch recreation of a paper for a class

## Hyperparameter Tuning
For the most part, I closely follow the recipe given in the original paper.  I did do an experiment with training for 200 epochs vs 100 epochs on the 500-sample, 25-to-34-ellipse set, with scheduler decay adjusted accordingly (see logs/2022-04-28-17-53-58).  In this case, the model has an L1 loss of 7.38e-2 on the validation set at epoch 2 (epoch 1 is just a filler value) and achieves a minimum of 4.6447e-3 at epoch 194.  This represents 93.7% reduction in L1 loss. The model trained on the same data for 100 epochs and appropriate scheduler decay (see logs/2022-04-28-16-29-12) has an L1 validation loss of 7.30e-2 at epoch 2 and achieved a minimum loss of 6.0423e-3 at epoch 98.  This represents a 91.7% reduction in L1 loss.

Given the fact that the improvement after twice the effort was only 2% (not to mention test set statistics), I considered 100 epochs to be a good stopping point. 
## Acknowledgements
Thanks to Eric O Lebigot for his stack overflow suggestion on drawin in NumPy arrays. https://stackoverflow.com/questions/25050899/producing-an-array-from-an-ellipse
Thanks to Saylor Academy for their 2012 Intermediate Algebra book, where I consulted their equations for an ellipse in standard form. https://saylordotorg.github.io/text_intermediate-algebra/s11-03-ellipses.html
