
package moaeval;

import moa.classifiers.Classifier;
import moa.classifiers.meta.LeveragingBag;
import com.yahoo.labs.samoa.instances.Instance;
import moa.streams.ArffFileStream;
import java.io.IOException;


public class eval 
{
    public int get_instance_class(Instance next_instance)
    {
        String a = next_instance.toString();
        int len_a = a.length();
        return Integer.parseInt(a.substring(len_a-4,len_a-3));
    }
    public void holdout_evaluate(ArffFileStream train_stream, ArffFileStream test_stream,Classifier classifier)
    {
        Instance next_instance;
        int tp=0,tn=0,fp=0,fn=0,instance_class;
        boolean positive_instace;
        while(train_stream.hasMoreInstances())
        {
            next_instance = train_stream.nextInstance().getData();
            classifier.trainOnInstance(next_instance);
        }
        while(test_stream.hasMoreInstances())
        {
            next_instance = test_stream.nextInstance().getData();
            instance_class = get_instance_class(next_instance);
            if(instance_class == 1)
                positive_instace = true;
            else
                positive_instace = false;
            if(classifier.correctlyClassifies(next_instance))
            {
                if(positive_instace)
                    tp++;
                else
                    tn++;
            }
            else
                if(positive_instace)
                    fn++;
                else
                    fp++;            
        }
        System.out.print(String.format("%d,%d,%d,%d",tp,tn,fp,fn));
    }

    public void holdout_evaluate_LeveragingBag(String train_file,String test_file)
    {
        ArffFileStream train_stream = new ArffFileStream(train_file,-1);
        train_stream.prepareForUse();
        ArffFileStream test_stream = new ArffFileStream(test_file,-1);
        test_stream.prepareForUse();

        LeveragingBag classifier = new LeveragingBag();
        classifier.ensembleSizeOption.setValue(200);;
        classifier.setModelContext(train_stream.getHeader());
        classifier.prepareForUse();

        holdout_evaluate(train_stream,test_stream,classifier);
    }


public static void main(String[] args) throws IOException
{
    eval exp = new eval();
    String train_file = args[0];
    String test_file = args[1];

    exp.holdout_evaluate_LeveragingBag(train_file,test_file);
}
}
