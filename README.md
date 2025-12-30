This project contains my attempts to construct a neural flux function for low accuracy solutions for Conservation laws, common in fluid mechanics.

Current status: Broken. The repo contains the final model "FirstIterate", and phaseComplete.ipynb within hybrid_pinn_fvm contains the entire training code. Due to changes made trying to extend the model, phase 3-6 are not
functional right now. Phase 1, 2 and 7 are. Phase 1 and 2 are the data generation phases, and phase 7 is testing. The final model can be loaded into the ipynb and tested in Phase 7. We can also generate new data in Phase 1/2 for testing.
