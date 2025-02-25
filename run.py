import Trainer
import DatasetClass
import Predictor


def main(ags.argparser):
    datasetconstructor = DatasetClass.DatasetConstructor()
    trainer = Trainer()
    predictor = Predictor()
    dataset, n_events = datasetconstructor.buildDataset()
    train_data, train_target ,test_data, test_target = trainer.splitDataset(dataset)

    model = trainer.buildModel(args.model)
    model.trainModel(train_data)
    
    pred_target = predictor.predict(test_data)
    rmse_test = model.evalModel(pred_target, test_target)
    rmse_train = model.evelModel(pred_target, train_target)



if __name__ == "__main__":
    main()


    


