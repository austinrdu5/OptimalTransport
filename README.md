# Quadratically-Regularized Wasserstein Distance on Graphs

This repository contains code and experiments for the paper "Quadratically-Regularized Wasserstein Distance on Graphs". 
The paper explores efficient methods of estimating the QR-Wasserstein distance between mass distributions on a graph.

## Introduction

The study of optimal transport attempts to find the most efficient method of transforming an initial mass distribution to a 
target mass distribution, given various constraints. This efficiency is measured via a “cost function” that places weights on 
the individual aspects of each transport, resulting in the “Wasserstein distance” or “earth-mover’s distance”. In doing so, one 
quantifies the difference between distributions, allowing classification or clustering of such distributions. 

This repository provides code and experiments for two algorithms that estimate the QR-Wasserstein distance between mass 
distributions on a graph. The algorithms use random connectivity and a novel geometric approach respectively to bypass the 
computationally intense quadratically-regularized (QR) optimal transport required for calculating QR-Wasserstein distance.

