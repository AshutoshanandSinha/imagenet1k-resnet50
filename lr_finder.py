import torch
from torch_lr_finder import LRFinder
import matplotlib.pyplot as plt
from pathlib import Path

def find_optimal_lr(model, train_loader, criterion, optimizer, device, 
                   num_iter=100, start_lr=1e-7, end_lr=10):
    """
    Find the optimal learning rate using the learning rate finder.
    """
    # Initialize the learning rate finder
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    
    # Run the range test
    lr_finder.range_test(train_loader, start_lr=start_lr, end_lr=end_lr, 
                        num_iter=num_iter, step_mode="exp")
    
    # Get suggestion for max_lr
    suggested_lr = lr_finder.suggestion()
    
    # Plot the loss vs learning rate
    fig, ax = plt.subplots()
    lr_finder.plot(suggest=True)
    
    # Save the plot
    Path('lr_finder_results').mkdir(exist_ok=True)
    plt.savefig('lr_finder_results/lr_finder_plot.png')
    plt.close()
    
    # Reset the model and optimizer to their initial state
    lr_finder.reset()
    
    return suggested_lr