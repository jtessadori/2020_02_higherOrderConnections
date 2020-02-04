classdef hoc < handle
    % January 2020, Jacopo Tessadori
    % The idea is to verify whether higher order connections are useful in
    % discriminating between healthy and RR subjects: in theory, if AB, BC
    % and CA are often active togther in a particular subject but one of
    % the connections is systematically missing in the average of healthy
    % controls, it is likely that that subject is re-routing a connection
    % to compensate for a missing link
    
    properties
        dataPath;
        FC;
        lbls;
        log;
        nROIs;
        feats;
    end
    
    methods
        function this=hoc(path)
            % Constructor for hoc class.
            % Only argument (required) is absolut path of data file.
            this.dataPath=path;
        end
        
        function loadData(this)
            % Recovers dynamical functional connectivity matrices as well
            % as lbls from the file specified as dataPath
            inData=load(this.dataPath);
            
            % Old datasets may have data split in different ways
            if isfield(inData,'CM')
                if length(inData.CM)==28
                    % This is a very specific test dataset, define lbls
                    % here
                    this.lbls=zeros(1,28);
                    this.lbls(15:end)=1;
                    this.FC=inData.CM;
                end
            else
                % Define labels and remove unlabeled data
                this.lbls=cat(1,zeros(length(inData.idHC),1),ones(length(inData.idRR),1));
                inData.cm_all_subj_corr=inData.cm_all_subj_corr(union(inData.idHC,inData.idRR));
                this.FC=inData.cm_all_subj_corr;
            end
%             
%             % Convert all covariance matrices into correlation matrices
%             for currSubj=1:length(this.FC)
%                 for currSlice=1:size(this.FC{currSubj},3)
%                     D=sqrt(diag(diag(squeeze(this.FC{currSubj}(:,:,currSlice)))));
%                     Dinv=pinv(D);
%                     this.FC{currSubj}(:,:,currSlice)=Dinv*this.FC{currSubj}(:,:,currSlice)*Dinv;
%                 end
%             end
        end
        
        function computeOutlyingConnections(this)
            for currSubj=1:length(this.lbls)
                this.log{currSubj}=[];
                
                % Compute three-way coactivations for current subject
                sData=abs(this.FC{currSubj});
                sa=repmat(permute(sData,[4 1 2 3]),this.nROIs,1,1,1);
                sb=repmat(permute(sData,[1 4 2 3]),1,this.nROIs,1,1);
                sc=repmat(permute(sData,[1 2 4 3]),1,1,this.nROIs,1);
                sHOC=sa.*sb.*sc;
                
                % Compute 'sham' three-way coactivations to determine
                % threshold
                shamData=abs(reshape(sData(randperm(numel(sData))),size(sData,1),size(sData,2),size(sData,3)));
                sa=repmat(permute(shamData,[4 1 2 3]),this.nROIs,1,1,1);
                sb=repmat(permute(shamData,[1 4 2 3]),1,this.nROIs,1,1);
                sc=repmat(permute(shamData,[1 2 4 3]),1,1,this.nROIs,1);
                shamHOCavg=squeeze(mean(sa.*sb.*sc,4));
                th=max(reshape(shamHOCavg,[],1));
%                 th=0.3;
                
                % Recover data for healthy subjects, excluding current one, if
                % relevant
                hSubjsIdx=find(this.lbls==0);
                hData=cat(3,this.FC{setdiff(hSubjsIdx,currSubj)});
                
                % Define average 3-way coactivations above a
                % certain threshold as "often active together"
                sHOCavg=squeeze(mean(sHOC,4));
                actIdx=find(sHOCavg>th);
                [idx1,idx2,idx3]=ind2sub(size(sHOCavg),actIdx);
                
                % Remove all entries with a duplicate coord
                actSubs=[idx1,idx2,idx3];
                actSubs(cellfun(@(x)length(unique(x))~=3,num2cell(actSubs,2)),:)=[]; % Removing entries with dublicate coords
                actSubs=unique(sort(actSubs,2),'rows'); % Can do this, as order is irrelevant (matrix is VERY symmetric)
                
%                 % Determine which connections are significant in healthy
%                 % subject data
%                 hMat=nan(nROIs);
%                 for currDim1=1:nROIs
%                     for currDim2=currDim1:nROIs
%                         [~,hMat(currDim1,currDim2)]=signrank(squeeze(hData(currDim1,currDim2,:)),[],'tail','right','alpha',.05/((nROIs*nROIs)/2));
%                         hMat(currDim2,currDim1)=hMat(currDim1,currDim2);
%                     end
%                 end
                % Using median is an approximation, but much faster
                hMat=squeeze(median(hData,3))>0;
                
                % For each strong 3-way coactivation, I need to test whether
                % exactly one strong connection exists on healthy subjects
                for currSub=1:size(actSubs,1)
                    hVec=[hMat(actSubs(currSub,1),actSubs(currSub,2)),hMat(actSubs(currSub,1),actSubs(currSub,3)),hMat(actSubs(currSub,2),actSubs(currSub,3))];
                    if sum(hVec)==1 % i.e. only one connection is significantly larger than 0
                        % I found a likely re-route. Shuffling order so
                        % that only link in healthy subjects is always
                        % between first two elements
                        switch hVec*(1:3)' % Case 1 requires no changes
                            case 2 % i.e. 1-3
                                actSubs(currSub,:)=actSubs(currSub,[1 3 2]);
                            case 3 % i.e. 2-3
                                actSubs(currSub,:)=actSubs(currSub,[2 3 1]);
                        end
                        
                        % I found a likely re-route, let's log it
                        if isempty(this.log{currSubj})
                            this.log{currSubj}=actSubs(currSub,:);
                        else
                            this.log{currSubj}=cat(1,this.log{currSubj},actSubs(currSub,:));
                        end
                    end
                end
                
                % Print progress
                fprintf('%d/%d subj lbl: %d, # re-routes: %d\n',currSubj,length(this.lbls),this.lbls(currSubj),size(this.log{currSubj},1));
            end
        end
        
        function recoverFeatures(this)
            % Not sure how to classify data with variable number of
            % elements. At the moment, I will try transforming data so that
            % I obtain a distribution of substituted and substituting links
            % for each subject and hope that works
            this.feats=zeros(this.nROIs^3,length(this.lbls));
            for currSubj=1:length(this.lbls)
                currSubjIdx=this.log{currSubj}*[this.nROIs^2,this.nROIs,1]';
                this.feats(currSubjIdx,currSubj)=1;
            end
            
            % Remove triplets that never occur
            this.feats(sum(this.feats,2)==0,:)=[];
        end
        
        function BAcc=testClassifier(this)
            % Recover labels proportions
            pdf=histcounts(this.lbls,length(unique(this.lbls)),'Normalization','probability');
            costMat=[0,1/pdf(1);1/pdf(2),0];
            
            % Determine best classifier structure, using half data for
            % training and half for validation (this causes some degree of
            % double dipping with the next step)
            BAccFun=@(Yreal,Yest)((sum((Yreal==0).*(Yest==0))/sum(Yreal==0))+(sum((Yreal==1).*(Yest==1))/sum(Yreal==1)))/2;
%             %% Random forest (or is it?)
%             errFun=@(x)1-BAccFun(this.lbls(2:2:end),...
%                 predict(fitcensemble(this.feats(:,1:2:end)',this.lbls(1:2:end)',...
%                 'Cost',costMat,'NumLearningCycles',x.nCycles,'Method',char(x.Method),'Cost',costMat,...
%                 'Learners',templateTree('MaxNumSplits',x.MaxNumSplits,'MinLeafSize',x.nLeaves)),this.feats(:,2:2:end)'));
%             optCycles=optimizableVariable('nCycles',[20 500],'Type','integer');
%             optLeaves=optimizableVariable('nLeaves',[1 100],'Type','integer');
%             optMethod=optimizableVariable('Method',{'Bag', 'GentleBoost', 'LogitBoost', 'RUSBoost'});
%             optMaxSplits=optimizableVariable('MaxNumSplits',[2 100],'Type','integer');
%             results=bayesopt(errFun,[optCycles,optLeaves,optMethod,optMaxSplits],...
%                 'AcquisitionFunctionName','expected-improvement-plus','MaxObjectiveEvaluations',100);%,'Verbose',0,'PlotFcn',{});
%                 
%             % Train cross-validated classifier
%             ens=fitcensemble(this.feats',this.lbls','KFold',5,'Cost',costMat,'Method',char(results.XAtMinEstimatedObjective.Method),...
%                 'NumLearningCycles',results.XAtMinEstimatedObjective.nCycles,'Learners',templateTree('MaxNumSplits',...
%                 results.XAtMinEstimatedObjective.MaxNumSplits,'MinLeafSize',results.XAtMinEstimatedObjective.nLeaves));
%             
%             % Recover predictions
%             lblsEst=ens.kfoldPredict;
            
            %%  Gauss SVM
            errFun=@(x)1-BAccFun(this.lbls(2:2:end),...
                predict(fitcsvm(this.feats(:,1:2:end)',this.lbls(1:2:end)',...
                'Cost',costMat,'BoxConstraint',x.BC,'KernelScale',x.KS,'Cost',costMat,...
                'KernelFunction','gaussian'),this.feats(:,2:2:end)'));
            optBC=optimizableVariable('BC',[1e-7,1e7],'Transform','log');
            optKS=optimizableVariable('KS',[1e-3,1e3],'Transform','log');
            results=bayesopt(errFun,[optBC,optKS],...
                'AcquisitionFunctionName','expected-improvement-plus','MaxObjectiveEvaluations',100);%,'Verbose',0,'PlotFcn',{});
            
            % Train cross-validated classifier
            svmMdl=fitcsvm(this.feats',this.lbls','KFold',5,'Cost',costMat,'KernelFunction','gaussian',...
                'KernelScale',results.XAtMinEstimatedObjective.KS,...
                'BoxConstraint',results.XAtMinEstimatedObjective.BC);
            
            % Recover predictions
            lblsEst=svmMdl.kfoldPredict;            

%             %%  Linear SVM (best so far)
%             errFun=@(x)1-BAccFun(this.lbls(2:2:end),...
%                 predict(fitcsvm(this.feats(:,1:2:end)',this.lbls(1:2:end)',...
%                 'Cost',costMat,'BoxConstraint',x.BC,'Cost',costMat,...
%                 'KernelFunction','linear'),this.feats(:,2:2:end)'));
%             optBC=optimizableVariable('BC',[1e-4,1e4],'Transform','log');
%             results=bayesopt(errFun,optBC,...
%                 'AcquisitionFunctionName','expected-improvement-plus','MaxObjectiveEvaluations',100);%,'Verbose',0,'PlotFcn',{});
%             
%             % Train cross-validated classifier
%             svmMdl=fitcsvm(this.feats',this.lbls','KFold',5,'Cost',costMat,'KernelFunction','linear',...
%                 'BoxConstraint',results.XAtMinEstimatedObjective.BC);
%             
%             % Recover predictions
%             lblsEst=svmMdl.kfoldPredict;
            
%             %%  Naive Bayes (doesn't seem to be working at all)
%             errFun=@(x)1-BAccFun(this.lbls(2:2:end),...
%                 predict(fitcnb(this.feats(:,1:2:end)',this.lbls(1:2:end)',...
%                 'Cost',costMat,'Width',x.K,'Cost',costMat,...
%                 'DistributionNames','kernel'),this.feats(:,2:2:end)'));
%             optK=optimizableVariable('K',[1e-4,1e4],'Transform','log');
%             results=bayesopt(errFun,optK,...
%                 'AcquisitionFunctionName','expected-improvement-plus','MaxObjectiveEvaluations',100);%,'Verbose',0,'PlotFcn',{});
%             
%             % Train cross-validated classifier
%             nbMdl=fitcnb(this.feats',this.lbls','KFold',5,'Cost',costMat,'DistributionNames','kernel',...
%                 'Width',results.XAtMinEstimatedObjective.K);
%             
%             % Recover predictions
%             lblsEst=nbMdl.kfoldPredict;
            
%             %% Poly-SVM
%             errFun=@(x)1-BAccFun(this.lbls(2:2:end),...
%                 predict(fitcsvm(this.feats(:,1:2:end)',this.lbls(1:2:end)',...
%                 'Cost',costMat,'BoxConstraint',x.BC,'Cost',costMat,...
%                 'PolynomialOrder',x.PO,'KernelFunction','polynomial'),this.feats(:,2:2:end)'));
%             optBC=optimizableVariable('BC',[1e-4,1e4],'Transform','log');
%             optPO=optimizableVariable('PO',[2 4],'Type','integer');
%             results=bayesopt(errFun,[optBC,optPO],...
%                 'AcquisitionFunctionName','expected-improvement-plus','MaxObjectiveEvaluations',100);%,'Verbose',0,'PlotFcn',{});
%             
%             % Train cross-validated classifier
%             svmMdl=fitcsvm(this.feats',this.lbls','KFold',5,'Cost',costMat,'KernelFunction','polynomial',...
%                 'BoxConstraint',results.XAtMinEstimatedObjective.BC,...
%                 'PolynomialOrder',results.XAtMinEstimatedObjective.PO);
%             
%             % Recover predictions
%             lblsEst=svmMdl.kfoldPredict;
            
            % Compute balanced accuracy
            BAcc=BAccFun(this.lbls,lblsEst);
        end
    
function n=get.nROIs(this)
    n=size(this.FC{1},1);
end
    end
end