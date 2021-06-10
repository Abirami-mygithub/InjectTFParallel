# InjectTFParallel


> InjectTFParallel is a fault injection software for simulating hardware faults to evaluate the reliability of non-sequential deep learning models (i.e Deep learning models with branches). <br />
For example: [[ResNet50]](#1), [[InceptionV3]](#2), [[MobileNetV2]](#3)
---

### Table of Contents

- [Description](#description)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [How to use](#how-to-use)
- [Results](#results)
- [Implementation Details](#implementation-details)
- [References](#references)
- [License](#license)
- [Author Info](#author-info)
- [Acknowledgements](#acknowledgements)

---

## Description
InjectTFParallel software provides us fault injection functionality to test the resilience of deep learning models against hardware faults. [InjectTF2](https://github.com/mbsa-tud/InjectTF2) is a fault injection software for sequential models. InjectTFParallel mainly focuses on injecting fault into non-sequential models (i.e) model containing layers with more than one input or one output.
In InjectTFParallel, fault injection into a model is done by creating a duplicate copy of the model and inserting fault injection custom layers. Two types of faults namely random bit fault and specific bit fault can be injected using InjectTFParallel. Each of these faults have unique custom layer implementation. Based on the chosen fault type, appropriate custom layer is inserted into duplicate copy of the model(also called as fault model). Fault model is then evaluated on the test dataset.

Keract library is used to visualize intermediate layer outputs in order to understand the behaviour of the model on fault injection.

[Back To The Top](#InjectTFParallel)

---

## Getting Started

### Prerequisites
- Python (v3.x.x)

### Installation
- Clone the repository using following command.
```
$ git clone https://github.com/Abirami-mygithub/InjectTFParallel.git
```
- Create a virtual environment inside the cloned repository folder (InjectTFParallel).
```
$ python3 -m venv .venv
```
- Activate the created virtual environment
```
$ source .venv/bin/activate
```
- Install the dependencies from requirements.txt
```
$ pip install -r requirements.txt --no-cache-dir
```
---
### How to use

1. Import fault injector component
```
 from injecttf.fault_injector import Fault_Injector
```
2. Instantiate fault injector component
```
 fi_obj = Fault_Injector()
```
3. Call inject_fault() from fault injector component using the instantiated object.
```
 fi_obj.inject_fault()
```
[example.py](https://github.com/Abirami-mygithub/InjectTFParallel/blob/main/injecttf/example.py) is created to show the usage of InjectTFParallel fault injector.

---

## Implementation Details
InjectTFParallel software architecture mainly comprises of configuration manager, model trainer, fault injector, model creator, datasets, custom layer, evaluation and visualization components. Each of these components have individual responsibility and interact with each other to provide the fault injection functionality. 
- **Configuration Manager:** It is responsible for reading from the [config file](https://github.com/Abirami-mygithub/InjectTFParallel/blob/main/config/Fault_injection_config_file.yml) containing fault injection specification and provide the fault injection configuration on request. It provides the config data in python dictionary.

- **[Model Creator:](https://github.com/Abirami-mygithub/InjectTFParallel/blob/main/model_creator/model_inception.py)** It provides the model architecture based on the user's request .

- **Dataset:** It provides preprocessed mnist and gtsrb dataset 

- **[Model Trainer:](https://github.com/Abirami-mygithub/InjectTFParallel/blob/main/model_trainer/trainer_inception.py)** It fetches the model from model creator and dataset, trains the model if not saved weights are available. It provides model and the weights .

- **Custom Layer:** Fault injection is done through creating custom layers. Two different custom layers namely [Fault_Injector_Random_Bit](https://github.com/Abirami-mygithub/InjectTFParallel/blob/main/custom_layer/fault_injection_random_bit.py) and [Fault_Injector_Specific_Bit](https://github.com/Abirami-mygithub/InjectTFParallel/blob/main/custom_layer/fault_injection_specific_bit.py) is created. Each has its own functionality.

- **[Fault Injector:](https://github.com/Abirami-mygithub/InjectTFParallel/blob/main/injecttf/fault_injector.py)** It is responsible for creating a duplicate copy of the model for which fault is to be injected and based on the fault injection specification from config file, inserts appropriate fault injection custom layer into the duplicate copy of the model. Thereby a new fault model is created with fault injection custom layer. This fault model is compiled and evaluated.


#### Software Architecture

![Software Architecture](https://github.com/Abirami-mygithub/InjectTFParallel/blob/main/Project_Images/architecture.png)



#### Sequence Diagram

![Sequence Diagram](https://github.com/Abirami-mygithub/InjectTFParallel/blob/main/Project_Images/sequence%20diagram.png)

[Back To The Top](#InjectTFParallel)

---
## Results
#### Model architecture without fault
![Model architecture without fault]()
#### Model architecture with fault
![Model architecture with fault]()

#### Fault Type - Specific bit

#### Fault Type - Random bit

[Back To The Top](#InjectTFParallel)

---
## Possible Issues and Solutions

1. Error while loading the weights from .h5 file <br />
Downgrade h5py to < 3.0.0 by following command:
```
$ pip install 'h5py<3.0.0'
```
---
## References
<div style="text-align: justify">
<a id="1">[1]</a>  He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016. <br />
<a id="2">[2]</a>  Szegedy, Christian, et al. "Rethinking the inception architecture for computer vision." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016. <br />
<a id="3">[3]</a>  Sandler, Mark, et al. "Mobilenetv2: Inverted residuals and linear bottlenecks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.
</div>


[Back To The Top](#InjectTFParallel)

---

## License

MIT License

Copyright (c) 2021 Abirami Ravi

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

---

## Author Info

- [Abirami Ravi](https://www.linkedin.com/in/abiramiravi/)


## Acknowledgements

- [Herr Jun.-Prof. Dr.-Ing. Andrey Morozov](https://www.ias.uni-stuttgart.de/institut/team/Morozov/)

- [Sheng Ding](https://www.ias.uni-stuttgart.de/en/institute/team/Ding/)

- Special thanks to [Saiteja Malyala](https://github.com/saitejamalyala)

- [Keract](https://github.com/philipperemy/keract/blob/master/README.md)

[Back To The Top](#read-me-template)

---
