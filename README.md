<div align="left">
  <img src="./docs/logo-xtddft.jpg" height="80px"/>
</div>

Spin-adapted time-dependent density functional theory for open-shell systems
-----------------------------------------------

## Available methods

1. Spin-adapted TD-DFT for open-shell systems
  - SF-TDA for spin-flip-up excitations ($S_f=S_i+1$ for $S_i>=0$)
  - X-TDA for spin-conserving excitations ($S_f=S_i$ for $S_i>=1/2$)
  - XSF-TDA for spin-flip-down excitations ($S_f=S_i-1$ for $S_i>=1$)

2. Simplified approximation for X-TDA

3. State interaction for treating spin-orbit couplings
 
## Under developments

1. GPU acceleration

2. Gradients for excited state geometry optimization

3. Non-adiabatic couplings for non-adiabatic dynamics

## How to cite

When using XTDDFT, please cite

```bash

@article{li2010spin,
  title={Spin-adapted open-shell random phase approximation and time-dependent density functional theory. I. Theory},
  author={Li, Zhendong and Liu, Wenjian},
  journal={The Journal of chemical physics},
  volume={133},
  number={6},
  year={2010},
  publisher={AIP Publishing}
}

@article{zhao2025spin,
  title={Spin-adapted open-shell time-dependent density functional theory: towards a simple and accurate method for spin-flip-down excitations},
  author={Zhao, Hewang and Li, Zhendong},
  journal={arXiv preprint arXiv:2511.16906},
  year={2025}
}

@article{li2013combining,
  title={Combining spin-adapted open-shell TD-DFT with spin--orbit coupling},
  author={Li, Zhendong and Suo, Bingbing and Zhang, Yong and Xiao, Yunlong and Liu, Wenjian},
  journal={Molecular Physics},
  volume={111},
  number={24},
  pages={3741--3755},
  year={2013},
  publisher={Taylor \& Francis}
}
```

## License

[Apache License 2.0](https://github.com/Quantum-Chemistry-Group-BNU/PyNQS/blob/main/LICENSE)
