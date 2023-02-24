package uspace.djl.tutorial.customdataset.andgate;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import java.io.IOException;

public class TestFit {	 
	static void printDatasetInfo(Dataset dataset) throws IOException, TranslateException
	{
        for (Batch batch : dataset.getData(NDManager.newBaseManager())) {
			NDList data = batch.getData();
			System.out.println("batch.shape = "+data.getShapes()[0]); 
			System.out.println(data);
			for (NDArray nda: data)
			{
				System.out.println("NDArray = "+nda);
			}
			batch.close(); 
        }		
	}
	
	public static void main(String[] args) throws IOException, TranslateException 
	{		
		long inputSize = 2;
		long outputSize = 1;
		
		SequentialBlock block = new SequentialBlock();		
		block.add(Blocks.batchFlattenBlock(inputSize));
		block.add(Linear.builder().setUnits(outputSize).build());
		
		Model model = Model.newInstance("and");
		model.setBlock(block); 		
		
		DefaultTrainingConfig config = new DefaultTrainingConfig(Loss.sigmoidBinaryCrossEntropyLoss())
			.addEvaluator(new Accuracy()) 
			.addTrainingListeners(TrainingListener.Defaults.logging());

		int batchSize = 10; 
		Trainer trainer = model.newTrainer(config);
		trainer.initialize(new Shape(batchSize,2));
		
		Dataset dataset = new AndGateDataset.Builder().build(); 
		dataset.prepare(new ProgressBar()); 
		// printDatasetInfo(dataset); 
		
		int epoch = 2;
		System.out.println("pre fit epoch="+epoch);
		EasyTrain.fit(trainer, epoch, dataset, null);
	}	
}
