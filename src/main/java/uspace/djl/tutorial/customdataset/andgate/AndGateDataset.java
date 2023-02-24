package uspace.djl.tutorial.customdataset.andgate;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.dataset.RandomAccessDataset.BaseBuilder;
import ai.djl.training.dataset.Sampler;
import ai.djl.training.dataset.Record;
import ai.djl.util.Progress;
import java.io.IOException;

/**
 * AND gate custom dataset
 * 0,0 -> 0
 * 0,1 -> 0
 * 1,0 -> 0
 * 1,1 -> 1 
 */
public class AndGateDataset extends RandomAccessDataset {
	private float[][] data; 
	private float[][] labels; 

    public AndGateDataset(Builder builder) {
		super(builder); 
		System.out.println("new AndGateDataset()");
		this.data   = builder.data; 
		this.labels = builder.labels; 
    }
	
	@Override
	public Record get(NDManager manager, long index) throws IOException {
		System.out.println("AndGateDataset.get() index="+index);
		int i = Math.toIntExact(index); 
		NDArray datum = manager.create(this.data[i]);
		NDArray label = manager.create(this.labels[i]); 
		Record record = new Record(new NDList(datum), new NDList(label)); 
		return record; 
	}

	@Override
	protected long availableSize() {
		return this.data.length; 
	}
	
	public static final class Builder extends BaseBuilder<Builder> {
		public Builder() { 
			System.out.println("new Builder()");
		}
		float[][] data; 
		float[][] labels; 
		
		@Override
		protected Builder self() {
			return this; 
		}
		
		AndGateDataset build() throws IOException {
			System.out.println("Builder.build()");
			NDManager manager = NDManager.newBaseManager();
			this.data = new float[4][2]; 
			this.labels = new float[4][1]; 
			
			this.data[0] = new float[]{0,0};
			this.labels[0] = new float[]{0};
			
			this.data[1] = new float[]{1,0};
			this.labels[1] = new float[]{0};
			
			this.data[2] = new float[]{0,1};
			this.labels[2] = new float[]{0};
			
			this.data[3] = new float[]{1,1};
			this.labels[3] = new float[]{1};
			
			Sampler sampler = new AndSampler(this.data, this.labels); 
			this.setSampling(sampler);
			return new AndGateDataset(this); 
		}
	}
	
	@Override
	public void prepare(Progress prgrs) {		
	}
}
