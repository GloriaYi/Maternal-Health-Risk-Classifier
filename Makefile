.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

.PHONY: build
build: ## build the docker image from the Dockerfile
	docker build -t dockerlock --file Dockerfile .

.PHONY: up
up: ## stop and start docker-compose services
	# by default stop everything before re-creating
	make stop
	docker compose up -d

.PHONY: stop
stop: ## stop docker-compose services
	docker compose stop

.PHONY: remove
remove: ## remove docker-compose services
	docker compose rm

.PHONY: all clean

all: reports/health_analysis.html reports/health_analysis.pdf

# download and extract data
data/raw/maternal+health+risk.zip data/raw/Maternal\ Health\ Risk\ Data\ Set.csv: scripts/01_download_data.py
	python scripts/01_download_data.py \
		--url="https://archive.ics.uci.edu/static/public/863/maternal+health+risk.zip" \
		--write-to=data/raw

# validate data and log results
data/processed/validated_data.csv results/logs/validation_errors.log: data/raw/Maternal\ Health\ Risk\ Data\ Set.csv scripts/02_validate_data.py
	python scripts/02_validate_data.py \
		--raw-data=data/raw/Maternal\ Health\ Risk\ Data\ Set.csv \
		--data-to=data/processed \
		--log-to=results/logs

# split and preprocess data
data/processed/maternal_health_risk_train.csv data/processed/maternal_health_risk_test.csv results/models/maternal_risk_preprocessor.pickle: data/processed/validated_data.csv scripts/03_split_preprocess_data.py
	python scripts/03_split_preprocess_data.py \
		--validated-data=data/processed/validated_data.csv \
		--data-to=data/processed \
		--preprocessor-to=results/models \
		--test-size=0.3 \
		--random-state=123

# perform eda and save plots
results/figures/correlation_heatmap.png results/tables/feature_densities_by_risklevel.png: data/processed/maternal_health_risk_train.csv scripts/04_eda.py
	python scripts/04_eda.py \
		--processed-training-data=data/processed/maternal_health_risk_train.csv \
		--plot-to=results/figures \
		--tables-to=results/tables

# train model, create visualized tuning and save model and results
results/models/maternal_risk_classifier.pickle results/figures/svc_hyperparameter_tuning.png: data/processed/maternal_health_risk_train.csv results/models/maternal_risk_preprocessor.pickle scripts/05_fit_maternal_health_risk_classifier.py
	python scripts/05_fit_maternal_health_risk_classifier.py \
		--training-data=data/processed/maternal_health_risk_train.csv \
		--preprocessor=results/models/maternal_risk_preprocessor.pickle \
		--pipeline-to=results/models \
		--plot-to=results/figures \
		--seed=123

# evaluate model
results/tables/test_scores.csv results/tables/confusion_matrix.csv results/tables/auc_scores.csv results/figures/confusion_matrix.png results/figures/roc_curves.png: data/processed/maternal_health_risk_test.csv results/models/maternal_risk_classifier.pickle scripts/06_evaluate_maternal_health_risk_classifier.py
	python scripts/06_evaluate_maternal_health_risk_classifier.py \
		--processed-test-data=data/processed/maternal_health_risk_test.csv \
		--pipeline-from=results/models/maternal_risk_classifier.pickle \
		--plot-to=results/figures \
		--results-to=results/tables \
		--seed=123

# render reports
reports/health_analysis.html reports/health_analysis.pdf: reports/health_analysis.qmd \
reports/references.bib \
data/raw/Maternal\ Health\ Risk\ Data\ Set.csv \
data/processed/validated_data.csv \
results/logs/validation_errors.log \
data/processed/maternal_health_risk_train.csv \
data/processed/maternal_health_risk_test.csv \
results/models/maternal_risk_preprocessor.pickle \
results/figures/correlation_heatmap.png \
results/tables/feature_densities_by_risklevel.png \
results/models/maternal_risk_classifier.pickle \
results/figures/svc_hyperparameter_tuning.png \
results/tables/test_scores.csv \
results/tables/confusion_matrix.csv \
results/tables/auc_scores.csv \
results/figures/confusion_matrix.png \
results/figures/roc_curves.png
	quarto render reports/health_analysis.qmd --to html
	quarto render reports/health_analysis.qmd --to pdf

clean :
	rm -rf data/raw/* \
    rm -r results/models/maternal_risk_preprocessor.pickle \
		data/processed/maternal_health_risk_train.csv \
		data/processed/maternal_health_risk_test.csv \
		data/processed/scaled_maternal_health_risk_train.csv \
		data/processed/scaled_maternal_health_risk_test.csv \
        data/processed/validated_data.csv \
	rm -f results/figures/feature_densities_by_risklevel.png \
		results/figures/correlation_heatmap.png \
        results/figures/confusion_matrix.png \
        results/figures/roc_curves.png \
        results/figures/svc_hyperparameter_tuning.png \
	rm -f results/logs/validation_errors.log \
        results/models/maternal_risk_classfier.pickle \
	rm -f results/tables/auc_scores.csv \
		results/tables/confusion_matrix.csv \
        results/tables/test_scores.csv \
        results/tables/train_describe.csv \
        results/tables/train_info.txt \
    rm -rf reports/health_analysis.html \
		reports/health_analysis.pdf \
		reports/health_analysis_files