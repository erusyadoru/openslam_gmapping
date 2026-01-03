
#ifdef MACOSX
// This is to overcome a possible bug in Apple's GCC.
#define isnan(x) (x==FP_NAN)
#endif

/**Just scan match every single particle.
If the scan matching fails, the particle gets a default likelihood.*/
inline void GridSlamProcessor::scanMatch(const double* plainReading){
  // sample a new pose from each scan in the reference

  // Debug: log odom pose at scan match time
  if (m_infoStream){
    m_infoStream << "[ScanMatch] odom_pose=(" << m_odoPose.x << ", " << m_odoPose.y << ", " << m_odoPose.theta << ")" << std::endl;
    m_infoStream << "[ScanMatch] last_pose=(" << m_lastPartPose.x << ", " << m_lastPartPose.y << ", " << m_lastPartPose.theta << ")" << std::endl;
  }

  double sumScore=0;
  int failCount=0;
  for (ParticleVector::iterator it=m_particles.begin(); it!=m_particles.end(); it++){
    OrientedPoint corrected;
    double score, l, s;
    score=m_matcher.optimize(corrected, it->map, it->pose, plainReading);
    //    it->pose=corrected;
    if (score>m_minimumScore){
      it->pose=corrected;
    } else {
      failCount++;
	if (m_infoStream && failCount <= 3){
	  m_infoStream << "Scan Matching Failed, using odometry. Likelihood=" << l <<std::endl;
	  m_infoStream << "lp:" << m_lastPartPose.x << " "  << m_lastPartPose.y << " "<< m_lastPartPose.theta <<std::endl;
	  m_infoStream << "op:" << m_odoPose.x << " " << m_odoPose.y << " "<< m_odoPose.theta <<std::endl;
	}
    }

    m_matcher.likelihoodAndScore(s, l, it->map, it->pose, plainReading);
    sumScore+=score;
    it->weight+=l;
    it->weightSum+=l;

    //set up the selective copy of the active area
    //by detaching the areas that will be updated
    // NOTE: computeActiveArea removed - it's called inside registerScan when needed
    //       The call here was wasted since resample calls invalidateActiveArea before registerScan
    m_matcher.invalidateActiveArea();
  }
  if (m_infoStream){
    m_infoStream << "[ScanMatch] AvgScore=" << sumScore/m_particles.size()
                 << " failCount=" << failCount << "/" << m_particles.size() << std::endl;
  }
}

inline void GridSlamProcessor::normalize(){
  //normalize the log m_weights
  double gain=1./(m_obsSigmaGain*m_particles.size());
  double lmax= -std::numeric_limits<double>::max();
  for (ParticleVector::iterator it=m_particles.begin(); it!=m_particles.end(); it++){
    lmax=it->weight>lmax?it->weight:lmax;
  }
  //cout << "!!!!!!!!!!! maxwaight= "<< lmax << endl;
  
  m_weights.clear();
  double wcum=0;
  m_neff=0;
  for (std::vector<Particle>::iterator it=m_particles.begin(); it!=m_particles.end(); it++){
    m_weights.push_back(exp(gain*(it->weight-lmax)));
    wcum+=m_weights.back();
    //cout << "l=" << it->weight<< endl;
  }
  
  m_neff=0;
  for (std::vector<double>::iterator it=m_weights.begin(); it!=m_weights.end(); it++){
    *it=*it/wcum;
    double w=*it;
    m_neff+=w*w;
  }
  m_neff=1./m_neff;
  
}

inline bool GridSlamProcessor::resample(const double* plainReading, int adaptSize, const RangeReading* reading){

  bool hasResampled = false;

  TNodeVector oldGeneration;
  for (unsigned int i=0; i<m_particles.size(); i++){
    oldGeneration.push_back(m_particles[i].node);
  }

  // Find best particle index for single-map mode
  int bestIdx = 0;
  if (m_singleMapMode) {
    double maxWeight = -std::numeric_limits<double>::max();
    for (unsigned int i = 0; i < m_particles.size(); i++) {
      if (m_particles[i].weight > maxWeight) {
        maxWeight = m_particles[i].weight;
        bestIdx = i;
      }
    }
  }

  if (m_neff<m_resampleThreshold*m_particles.size()){

    if (m_infoStream)
      m_infoStream  << "*************RESAMPLE***************" << std::endl;

    uniform_resampler<double, double> resampler;
    m_indexes=resampler.resampleIndexes(m_weights, adaptSize);

    if (m_outputStream.is_open()){
      m_outputStream << "RESAMPLE "<< m_indexes.size() << " ";
      for (std::vector<unsigned int>::const_iterator it=m_indexes.begin(); it!=m_indexes.end(); it++){
	m_outputStream << *it <<  " ";
      }
      m_outputStream << std::endl;
    }

    onResampleUpdate();

    if (m_singleMapMode) {
      // Single-map mode: skip particle copying, just update poses and register scan to best particle only
      std::cerr << "[SingleMapMode] Skipping particle copy, registering scan to best particle only..." << std::endl;

      // Update trajectory tree without copying particles
      for (unsigned int i=0; i<m_particles.size(); i++){
        TNode* node = new TNode(m_particles[i].pose, 0, oldGeneration[i], 0);
        node->reading = reading;
        m_particles[i].node = node;
        m_particles[i].setWeight(0);
        m_particles[i].previousIndex = i;
      }

      // Register scan only to the best particle's map
      m_matcher.invalidateActiveArea();
      m_matcher.registerScan(m_particles[bestIdx].map, m_particles[bestIdx].pose, plainReading);

      std::cerr << " Done (best=" << bestIdx << ")" << std::endl;
      hasResampled = true;

    } else {
      // Original behavior: copy particles with maps
      //BEGIN: BUILDING TREE
      ParticleVector temp;
      unsigned int j=0;
      std::vector<unsigned int> deletedParticles;

      for (unsigned int i=0; i<m_indexes.size(); i++){
        while(j<m_indexes[i]){
          deletedParticles.push_back(j);
          j++;
        }
        if (j==m_indexes[i])
          j++;
        Particle & p=m_particles[m_indexes[i]];
        TNode* node=0;
        TNode* oldNode=oldGeneration[m_indexes[i]];
        node=new TNode(p.pose, 0, oldNode, 0);
        node->reading=reading;

        temp.push_back(p);
        temp.back().node=node;
        temp.back().previousIndex=m_indexes[i];
      }
      while(j<m_indexes.size()){
        deletedParticles.push_back(j);
        j++;
      }

      std::cerr <<  "Deleting Nodes:";
      for (unsigned int i=0; i<deletedParticles.size(); i++){
        std::cerr <<" " << deletedParticles[i];
        delete m_particles[deletedParticles[i]].node;
        m_particles[deletedParticles[i]].node=0;
      }
      std::cerr  << " Done" <<std::endl;

      //END: BUILDING TREE
      std::cerr << "Deleting old particles..." ;
      m_particles.clear();
      std::cerr << "Done" << std::endl;
      std::cerr << "Copying Particles and  Registering  scans...";

      // Optimized: compute activeArea once, share across all particles
      ParticleVector::iterator first_it = temp.begin();
      first_it->setWeight(0);
      m_matcher.invalidateActiveArea();
      m_matcher.registerScan(first_it->map, first_it->pose, plainReading);
      m_particles.push_back(*first_it);

      // Get the computed activeArea from first particle's map
      const HierarchicalArray2D<PointAccumulator>::PointSet& sharedActiveArea =
          ScanMatcher::getActiveArea(first_it->map);

      // Remaining particles: use shared activeArea
      for (ParticleVector::iterator it = first_it + 1; it != temp.end(); it++){
        it->setWeight(0);
        m_matcher.registerScanWithActiveArea(it->map, it->pose, plainReading, sharedActiveArea);
        m_particles.push_back(*it);
      }
      std::cerr  << " Done" <<std::endl;
      hasResampled = true;
    }
  } else {
    // No resampling needed
    int index=0;
    std::cerr << "Registering Scans:";
    TNodeVector::iterator node_it=oldGeneration.begin();

    if (m_singleMapMode) {
      // Single-map mode: only register scan to best particle
      for (ParticleVector::iterator it=m_particles.begin(); it!=m_particles.end(); it++){
        TNode* node = new TNode(it->pose, 0.0, *node_it, 0);
        node->reading=reading;
        it->node=node;
        it->previousIndex=index;
        index++;
        node_it++;
      }
      // Register scan only to best particle
      m_matcher.invalidateActiveArea();
      m_matcher.registerScan(m_particles[bestIdx].map, m_particles[bestIdx].pose, plainReading);
      std::cerr << " Done (single-map, best=" << bestIdx << ")" << std::endl;

    } else {
      // Optimized: compute activeArea once, share across all particles
      // First particle: compute activeArea
      ParticleVector::iterator first_it = m_particles.begin();
      TNode* first_node = new TNode(first_it->pose, 0.0, *node_it, 0);
      first_node->reading = reading;
      first_it->node = first_node;
      m_matcher.invalidateActiveArea();
      m_matcher.registerScan(first_it->map, first_it->pose, plainReading);
      first_it->previousIndex = index;
      index++;
      node_it++;

      // Get the computed activeArea from first particle's map
      const HierarchicalArray2D<PointAccumulator>::PointSet& sharedActiveArea =
          ScanMatcher::getActiveArea(first_it->map);

      // Remaining particles: use shared activeArea
      for (ParticleVector::iterator it = first_it + 1; it != m_particles.end(); it++){
        TNode* node = new TNode(it->pose, 0.0, *node_it, 0);
        node->reading = reading;
        it->node = node;
        m_matcher.registerScanWithActiveArea(it->map, it->pose, plainReading, sharedActiveArea);
        it->previousIndex = index;
        index++;
        node_it++;
      }
      std::cerr  << "Done" <<std::endl;
    }
  }
  //END: BUILDING TREE

  return hasResampled;
}
