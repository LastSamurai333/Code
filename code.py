Error While creating retinanet_box model
**AttributeError: 'NoneType' object has no attribute '_inbound_nodes'**

Execution Sequence
retinanet_box--->filter_detection---->detections_logic -->nms_n_score_threshold

def nms_n_score_threshold(boxes, scores, labels, max_detections, score_threshold, iou_threshold):
  #Return indices contain indexes of Anchors after score_threshold+nms 
  
   
  #Apply score_threshold + nms
  indices = keras.layers.Lambda( lambda where:tf.where(K.greater(scores,score_threshold)))#Check #shape=[X,1], X=Not fixed=Reduced_anchors; give index value for (scores>threshold) e.g. [[1][2][3]]  X=True elements
  
  filtered_boxes=keras.layers.Lambda( lambda gather1: tf.gather_nd(boxes,indices))#Check #shape=[X,4]        # Reason to use gather_nd:-If we use gather here shape[X,1,4]
  filtered_scores=K.gather(scores,indices)[:,0]#shape=[X,1] --> [X]  #Reduced_anchors=Anchors whoes score > score_threshold
  
  #nms: shape=[M] ; M=Seleted Anchores/Tensors (M <= max_output_size) #Check
  nms_indices=keras.layers.Lambda( lambda image1:tf.image.non_max_suppression(boxes=filtered_boxes, scores=filtered_scores, max_output_size=max_detections, iou_threshold=iou_threshold)) #e.g. o/p=[1 10 3]
  
  indices = K.gather(indices,nms_indices) #[M,1]
  labels = keras.layers.Lambda( lambda gather2:tf.gather_nd(labels, indices)) #[M] #Check
  indices_labels = keras.backend.stack([indices[:, 0], labels], axis=1)#shape=[M,2]
  return indices_labels 
  
  
  
  
  
#~~~~~~~~~~~~~~~~~~~~detections_logic~~~~~~~~~~~~~~~~~~~#
def detections_logic(boxes,classification):
  max_detections=10
    if class_specific_filter: #classification=[Anchors,num_class]
        all_indices = []
        # perform per class filtering
        for c in range(int(classification.shape[1])):
            scores = classification[:, c] #shape=[some_anchors]
            labels = c * backend.ones((keras.backend.shape(scores)[0],), dtype='int64') #shape=[some_anchors]
            all_indices.append(nms_n_score_threshold(boxes, scores, labels,max_detections,score_threshold,iou_threshold))

        # concatenate indices to single tensor
        indices_labels = k.Concatenate(all_indices, axis=0)#[X,2] X>M #check
  else:
    scores        = K.max   (classification,axis=1) #shape=[some_anchors]
    labels        = K.argmax(classification,axis=1) #shape=[some_anchors]
    indices_labels= nms_n_score_threshold(boxes, scores, labels,max_detections,score_threshold,iou_threshold) #shape=[M,1]
    
    
    # Select Top k(max_detection)
    labels              = indices_labels[:, 1] #[X] or [M]
    scores   = keras.layers.Lambda( lambda gather3:tf.gather_nd(classification, indices_labels))#Check#[X] or [M]   #shape=[X] for class_specific_filter=TRUE
                                                                         #shape=[M] for class_specific_filter=FALSE
      
    scores_k, top_indices = keras.layers.Lambda( lambda top_k:(tf.top_k(input=scores, k=K.minimum(max_detections, K.shape(scores)[0]))))#Check #score=Rank 1, k= Rank 0  Tensor; top_k return value+indices
    #scores/top_indices = shape[k]
    
    # filter input using the final set of indices
    
    indices_k             = keras.backend.gather(indices_labels[:, 0], top_indices)#[K]
    boxes_k               = keras.backend.gather(boxes, indices_k)  #[K,4]   Where K <= Max_detection
    labels_k              = keras.backend.gather(labels, top_indices) #[K]
    
    # Zero pad at last rows
    pad_size = keras.backend.maximum(0, max_detections - K.shape(scores)[0])#PaddingSize= 0 or max_detection-K
    boxes    = keras.layers.Lambda( lambda pad1:tf.pad(boxes_k, [[0, pad_size], [0, 0]], constant_values=-1)) #shape[max_detections,4] #Check
    scores   = keras.layers.Lambda( lambda pad2:tf.pad(scores_k, [[0, pad_size]], constant_values=-1)) #shape[max_detections]
    labels   = keras.layers.Lambda( lambda pad3:tf.pad(labels_k, [[0, pad_size]], constant_values=-1)) #shape[max_detections]
    
    labels   = keras.backend.cast(labels, 'int32')
    boxes    = keras.backend.cast(boxes, keras.backend.floatx())
    scores   = keras.backend.cast(scores,keras.backend.floatx())
    
    #Verify Shape
    
    boxes.set_shape([max_detections, 4])
    scores.set_shape([max_detections])
    labels.set_shape([max_detections])
    
    return [boxes, scores, labels]
    
    
    
    
    
#~~~~~~~~~~~~~~~~filter_detection~~~~~~~~~~~~~~~~~~~~~~#    
class filter_detection(keras.layers.Layer):
  
  def compute_output_shape(self, input_shape):
    max_detections=10
    print("4")
    return [
        (input_shape[0][0], max_detections, 4),
        (input_shape[1][0], max_detections),
        (input_shape[1][0], max_detections)
    ] 
  
  
  
  
  def call(self,inputs):
    boxes          = inputs[0]
    classification = inputs[1]
    print("1")
    # wrap nms with our parameters
    def _detections_logic(args):
      boxes          = args[0]
      classification = args[1]
      print("2")

      return detections_logic(
          boxes,
          classification
      )
    outputs = keras.layers.Lambda( lambda map_fn:tf.map_fn(_detections_logic,elems=[boxes,classification],
                      dtype=[keras.backend.floatx(), keras.backend.floatx(),'int32'],
                      parallel_iterations=1 ))#Check
    print("3")
    print(outputs)    
    return outputs
    #dtype have 3 types for outputs of detections_logic ([boxes, scores, labels])




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~retinanet_box~~~~~~~~~~~~~~~~~~#
def retinanet_box(model,input_image_size,include_top, pyramid_network_filters,num_classes,
                  regression_filter,classification_filter,num_anchors,sizes,ratios,scales,strides,
                 class_specific_filter, max_detections,score_threshold,iou_threshold,parallel_iterations):
  if model is None:
    model = retinanet_model(input_image_size,include_top, pyramid_network_filters,num_classes,regression_filter,classification_filter,num_anchors)
  
  
  features = [model.get_layer(name=map).output for map in ['P3','P4','P5','P6','P7'] ]  #Pyramid feature map
  #print(features)
  anchors = generate_anchors(features,sizes,strides,ratios,scales,num_anchors) #[batch, Anchors, 4]; Entire anchors for all batch_images, all features, all pixels
  #print(anchors.get_shape())
  
  #Model output 
  regression    =model.outputs[0]   #Regression output= Return deltas factor of W(for x1,2) , H(for y1,y2); Shape=[batch,Anchors,4]
  classification=model.outputs[1]   #shape=[batch,Anchors,num_classes]
  print(regression)
  
  #Apply regression & clipped Anchors
  boxes=regressionbox(anchors,regression) #shape of anchors/regression = [batch,Anchors,4] 
  boxes=clipbox(boxes,model.inputs[0])    #model.input[0]=input/image shape=[W,H,3],  shape=[batch,Anchors,4]
  
  #Detection
  print("0")
  detections = (filter_detection()([boxes,classification]))
  print("5")
                
                #All inputs to the layer should be tensors so cant work ([boxes,classification,class_specific_filter,max_detections,score_threshold,iou_threshold,parallel_iterations]))
  print((detections))
  
  return keras.models.Model(inputs=model.inputs, outputs=detections, name='retinanet_box')
  
