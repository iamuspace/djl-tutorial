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
	private NDArray[] data; 
	private NDArray[] labels; 

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
		NDArray datum = this.data[i];
		NDArray label = this.labels[i]; 
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
		NDArray[] data; 
		NDArray[] labels; 
		
		@Override
		protected Builder self() {
			return this; 
		}
		
		AndGateDataset build() throws IOException {
			System.out.println("Builder.build()");
			NDManager manager = NDManager.newBaseManager();
			this.data = new NDArray[4]; 
			this.labels = new NDArray[4]; 
			
			NDArray data_0  = manager.create(new float[]{0,0});
			NDArray label_0 = manager.create(new float[]{0});
			this.data[0] = data_0; 
			this.labels[0] = label_0; 
			
			NDArray data_1  = manager.create(new float[]{1,0});
			NDArray label_1 = manager.create(new float[]{0});
			this.data[1] = data_1; 
			this.labels[1] = label_1; 
			
			NDArray data_2  = manager.create(new float[]{0,1});
			NDArray label_2 = manager.create(new float[]{0});
			this.data[2] = data_2; 
			this.labels[2] = label_2; 
			
			NDArray data_3  = manager.create(new float[]{1,1});
			NDArray label_3 = manager.create(new float[]{1});
			this.data[3] = data_3; 
			this.labels[3] = label_3; 
			
			Sampler sampler = new AndSampler(this.data, this.labels); 
			this.setSampling(sampler);
			return new AndGateDataset(this); 
		}
	}
	
	@Override
	public void prepare(Progress prgrs) {		
	}
}
