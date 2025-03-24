from src.piplines.training_pipeline import TrainingPipeline

# Initialize the training pipeline
config_path = 'config.yaml'
pipeline = TrainingPipeline(config_path)

# Run the training pipeline
if __name__ == '__main__':
    pipeline.run_training_pipeline()



