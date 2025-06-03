#!/bin/bash
python -m printerid.training.train --config config.yaml
python -m printerid.training.evaluate --config config.yaml
