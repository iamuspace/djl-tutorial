package uspace.djl.tutorial.customdataset.andgate;

import ai.djl.ndarray.NDArray;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.dataset.Sampler;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class AndSampler implements Sampler {
	private float[][] data; 
	private float[][] labels; 
	public AndSampler(float[][] data, float[][] labels)
	{
		System.out.println("new AndSampler()");
		this.data = data; 
		this.labels = labels; 
	}
	
	@Override
	public Iterator<List<Long>> sample(RandomAccessDataset rad) {
		System.out.println("AndSampler.sample() call rad size="+rad.size());
		
		List<List<Long>> list = new ArrayList(); 
		ArrayList<Long> batch1 = new ArrayList();
		batch1.add(0L);
		batch1.add(1L);
		batch1.add(2L); 
		batch1.add(3L); 
		list.add(batch1); 
		System.out.println("AndSampler.sample() list="+list);
		return list.iterator(); 
	}

	@Override
	public int getBatchSize() {
		return this.data.length; 
	}	
}
