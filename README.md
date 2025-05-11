# In-Context Learning for Structured Predictions

This repository contains the implementation code and evaluation notebooks for the paper "The Missing Alignment Link of In-context Learning on Sequences" (2025).

## Paper

The full paper can be accessed here: [2025_In_Context_Learning_for_Structured_Predictions.pdf](./2025_In_Context_Learning_for_Structured_Predictions.pdf)


## Repository Structure

- `code/`: Implementation of models and training procedures
  - `data.py`: Data loading and preprocessing utilities
  - `distribution.py`: Probability distribution modeling
  - `model.py`: Core model implementation
  - `pfa.py`: Preferential feature attention implementation
  - `training.py`: Training loops and optimization
  - `utils.py`: Helper functions and utilities

- `eval/`: Evaluation notebooks and analysis
  - `accuracy_metrics.ipynb`: Performance measurement across tasks
  - `attention_map.ipynb`: Visualization of attention patterns
  - `compare_models.ipynb`: Comparative analysis of different models
  - `heatmap.ipynb`: Heat map visualizations of model behavior

## License

This project is licensed under the MIT License - see the LICENSE file for details.

```
MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```