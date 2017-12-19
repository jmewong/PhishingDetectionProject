
include("./proxgrad.jl")

using DataFrames
using GZip

##possible types in arff files are:
#real, numeric,
#date
#string
#nominal specifications, starting with {

function convertType(attrType)
  x = lowercase(attrType)
  res = nothing
    
  if in(x, ["real" "numeric"])
    res = "numeric"
  elseif startswith(x, "{")
    res = "nominal"
  else #date and string
    res = x
  end
  return(res)
end

function readArff(filename, gzip=false)
  f = nothing
  if gzip
    f = GZip.gzopen(filename)
  else
    f = open(filename)
  end
  lines = eachline(f)
    
  header = true
  colNames = nothing
  attrTypes = nothing
  myData = DataFrame()
  attributes = DataFrame()
  nAttr = 0
  nData = 0

  for l in lines
    if header
      mAttr = match(r"^[[:space:]]*@(?i)attribute", l)
      if mAttr == nothing
        mData = ismatch(r"^[[:space:]]*@(?i)data", l)
        if mData
          colNames = convert(Array, attributes[2,:])
          attrTypes = map(convertType, convert(Array, attributes[3,:]))
          header=false
          break
        end
      else
        nAttr += 1
        attributes[:, nAttr] = split(l)[1:3]
      end
    end
  end
  myData = readtable(f, names = vec(map(Symbol, colNames)))
  close(f)
  return(myData, attrTypes)
end

dataset1, attributeTypes1 = readArff("Mohammad14JulyDS_1.arff");

dataset2, attributeTypes2 = readArff("Mohammad14JulyDS_2.arff");

dataset = vcat(dataset1, dataset2);

dataset = dataset[shuffle(1:end),:];

names(dataset)

# X = dataset[:, filter(x -> x != :Result, names(dataset))]
# [X[isna.(X[nm]), nm] =  0  for nm in names(X)]
# X = Array(X)

X = dataset[:, filter(x -> x != :Result, names(dataset))]
for nm in names(X)
    count = zeros(3)
    for entry in X[nm]
        if !isna.(entry)
            count[entry+2] += 1
        end
    end
    X[isna.(X[nm]), nm] = maximum(count)-2
    
end
X = Array(X)
y = convert(Array, dataset[:Result]);

y = convert(Array,dataset[:Result]);

Xtrain = X[1:4000,:]
Xtest = X[4001:end,:]
ytrain = y[1:4000]
ytest = y[4001:end];

using ScikitLearn: fit!, predict

Pkg.add("DecisionTree")

using ScikitLearn
using PyCall
using PyPlot
using ScikitLearn.CrossValidation: train_test_split
using DecisionTree

#using ScikitLearn.Models: DecisionTreeClassifier, RandomForestClassifier, AdaBoostStumpClassifier
@pyimport matplotlib.colors as mplc
@sk_import preprocessing: StandardScaler
@sk_import neighbors: KNeighborsClassifier
@sk_import svm: SVC
@sk_import naive_bayes: GaussianNB
@sk_import discriminant_analysis: (LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis)
using ScikitLearn.Utils: meshgrid

@sk_import linear_model: LogisticRegression

using ScikitLearn.GridSearch: GridSearchCV
gridsearch = GridSearchCV(LogisticRegression(penalty = "l1"), Dict(:C => 0.1:0.1:2.0))
fit!(gridsearch, Xtest, ytest)
println("Best parameters: $(gridsearch.best_params_)")

using ScikitLearn.GridSearch: GridSearchCV
gridsearch = GridSearchCV(LogisticRegression(penalty = "l2"), Dict(:C => 0.1:0.1:2.0))
fit!(gridsearch, Xtest, ytest)
println("Best parameters: $(gridsearch.best_params_)")

log = LogisticRegression(C = 0.5, penalty = "l1", fit_intercept=true, max_iter = 100)
fit!(log, Xtrain, ytrain)
#model0 = fit(model, X, y)
ylog = predict(log,Xtest) #the output vector of predictions
accuracy = count(ylog .== ytest)/length(ytest)
#accuracy = sum(predict(log, Xtest) .== ytest) / length(ytest)
#println("accuracy: $accuracy")

false_neg = 0 
false_pos = 0 
sum = 0 
for i=1:length(ylog)
    if ytest[i] != ylog[i]
        sum+=1
        if (ytest[i] == 1 && ylog[i] == -1)
            false_neg += 1
        else
            false_pos += 1
        end
    end
end
println(false_neg/length(ylog))
println(false_pos/length(ylog))

log2 = LogisticRegression(C = 0.1, penalty = "l2", fit_intercept=true)
fit!(log2, Xtrain, ytrain)
#model0 = fit(model, X, y)
ylog2 = predict(log2,Xtest) #the output vector of predictions
accuracy = count(ylog .== ytest)/length(ytest)
#accuracy = sum(predict(log, Xtest) .== ytest) / length(ytest)
#println("accuracy: $accuracy")


log

false_neg = 0 
false_pos = 0 
for i=1:length(ylog2)
    if ytest[i] != ylog2[i]
        sum+=1
        if (ytest[i] == 1 && ylog2[i] == -1)
            false_neg += 1
        else
            false_pos += 1
        end
    end
end
println(false_neg/length(ylog))
println(false_pos/length(ylog))

#the probabilities of either -1 (first column) or 1 (second column)
p = predict_proba(log, Xtest)

using ScikitLearn.CrossValidation: cross_val_score

@time cv1 = cross_val_score(LogisticRegression(C=0.5, penalty = "l1"), X, y; cv = 10)

using StatsBase, StatsFuns, StreamStats
summarystats(cv1)

@time cv2 = cross_val_score(LogisticRegression(C=0.1, penalty = "l2"), X, y; cv=10)

summarystats(cv2)

using LowRankModels

import LowRankModels: evaluate, grad
evaluate(loss::Loss, X::Array{Float64,2}, w, y) = evaluate(loss, X*w, y)
grad(loss::Loss, X::Array{Float64,2}, w, y) = X'*grad(loss, X*w, y)

# proximal gradient method
function proxgrad(loss::Loss, reg::Regularizer, X, y;
                  maxiters::Int = 10, stepsize::Number = 1., 
                  ch::ConvergenceHistory = ConvergenceHistory("proxgrad"))
    w = zeros(size(X,2))
    for t=1:maxiters
        t0 = time()
        # gradient step
        g = grad(loss, X, w, y)
        w = w - stepsize*g
        # prox step
        w = prox(reg, w, stepsize)
        # record objective value
        update_ch!(ch, time() - t0, obj = evaluate(loss, X, w, y) + evaluate(reg, w))
    end
    return w
end

reg = OneReg()
loss = LogisticLoss()

y_i = convert(Array{Bool}, ytrain .== 1);

X_i = convert(Array{Float64,2},Xtrain);

step = 1/norm(Xtrain)^2

using Plots

ch = ConvergenceHistory("OneReg")
w = proxgrad(loss, reg, X_i, y_i; 
             stepsize=step, maxiters=1000,
             ch = ch)

Plots.plot(ch.objective)
println(w)
println(maximum(abs.(w)))

Plots.plot(ch.objective)
xlabel!("iteration")
ylabel!("objective")

y_i2 = Xtest*w

ynew = []
for i in y_i2
    if i > 0
        append!(ynew,1)
    else
        append!(ynew,-1)
    end
end

sum = 0 
for i=1:length(ynew)
    if ynew[i] == ytest[i]
        sum+=1
    end
end
println(sum/length(ynew))

h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree (Julia)",
         "Random Forest (Julia)", "AdaBoost (Julia)", "Naive Bayes", "Linear Discriminant Analysis",
         "Quadratic Discriminant Analysis"]
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=5, C=4),
    DecisionTreeClassifier(pruning_purity_threshold=0.8),
    RandomForestClassifier(ntrees=30),
    # Note: scikit-learn's adaboostclassifier is better than DecisionTree.jl in this instance
    # because it's not restricted to stumps, and the data isn't axis-aligned
    AdaBoostStumpClassifier(niterations=30),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(), 
];

cvt = cross_val_score(DecisionTreeClassifier(pruning_purity_threshold=0.8), X, y; cv=10)

summarystats(cvt)

cvt = cross_val_score(RandomForestClassifier(ntrees=30),X,y; cv = 10)

summarystats(cvt)

for (name, clf) in zip(names, classifiers)
    fit!(clf, Xtrain, ytrain)
    scor = score(clf, Xtest, ytest)
    println(scor)
end

model = build_tree(ytrain, Xtrain)

model = prune_tree(model, 0.8)

print_tree(model,5)

ytree = apply_tree(model, Xtest); #apply model to test data

false_pos = 0
false_neg = 0 
sum = 0 
for i=1:length(ytree)
    if ytest[i] != ytree[i]
        sum+=1
        if (ytest[i] == 1 && ytree[i] == -1)
            false_neg += 1
        else
            false_pos += 1
        end
    end
end
println("false positive: ", false_pos/length(ytree))
println("false negative: ",false_neg/length(ytree))
println("total error ", sum/length(ytree))

accuracy = nfoldCV_forest(ytrain, Xtrain, 2, 100,3,0.5)

rf = build_forest(ytrain, Xtrain, 2, 100, 0.5, 50)

yforest = apply_forest(rf, Xtest);

false_pos = 0
false_neg = 0 
sum = 0 
for i=1:length(yforest)
    if ytest[i] != yforest[i]
        sum+=1
        if (ytest[i] == 1 && yforest[i] == -1)
            false_neg += 1
        else
            false_pos += 1
        end
    end
end
println("false positive: ", false_pos/length(yforest))
println("false negative: ",false_neg/length(yforest))
println("total error ", sum/length(yforest))
