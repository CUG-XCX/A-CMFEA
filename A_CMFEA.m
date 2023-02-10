classdef A_CMFEA < Algorithm
    % <Multi> <Constrained>

    % A-CMFEA 

    properties (SetAccess = private)
        rmp0 = 0.3
        mu = 2; % index of Simulated Binary Crossover (tunable)
        mum = 5; % index of polynomial mutation
        probswap = 0; % probability of variable swap
        beta=0.7;
        c=0.1
        ro=0
    end

    methods
        function parameter = getParameter(obj)
            parameter = {'rmp0: Random Mating Probability', num2str(obj.rmp0), ...
                        'mu: index of Simulated Binary Crossover (tunable)', num2str(obj.mu), ...
                        'mum: index of polynomial mutation', num2str(obj.mum), ...
                        'probSwap: Variable Swap Probability', num2str(obj.probswap),...
                        'beta',num2str(obj.beta),...
                        'c',num2str(obj.c),...
                        'ro',num2str(obj.ro)};
        end

        function obj = setParameter(obj, parameter_cell)
            count = 1;
            obj.rmp0 = str2double(parameter_cell{count}); count = count + 1;
            obj.mu = str2double(parameter_cell{count}); count = count + 1;
            obj.mum = str2double(parameter_cell{count}); count = count + 1;
            obj.probswap = str2double(parameter_cell{count}); count = count + 1;
            obj.beta = str2double(parameter_cell{count}); count = count + 1;
            obj.c = str2double(parameter_cell{count}); count = count + 1;
            obj.ro = str2double(parameter_cell{count}); count = count + 1;
        end

        function data = run(obj, Tasks, run_parameter_list)
            sub_pop = run_parameter_list(1);
            sub_eva = run_parameter_list(2);
            pop_size = sub_pop * length(Tasks);
            eva_num = sub_eva * length(Tasks);
            r=0;
            m=0;
            flag=0;
            c_flag=0;
            max_generation=eva_num/pop_size;
            archive=[];
            
            % initialize rmp
            for i=1:max_generation
                rmp(i)=obj.rmp0;
            end
           
            tic

            % initialize
   %        [population, fnceval_calls, bestobj, bestCV, data.bestX] = initializeMF_FP_1(Individual, pop_size, Tasks, length(Tasks));
            [population, fnceval_calls, bestobj, bestCV, bestX] = initializeCMF(Individual, pop_size, Tasks, max([Tasks.dims]));
            data.convergence(:, 1) = bestobj;
            data.convergence_cv(:, 1) = bestCV;
            % initialize affine transformation
            [mu_tasks, Sigma_tasks] = InitialDistribution(population, length(Tasks));

            generation = 1;
            while fnceval_calls < eva_num
                generation = generation + 1;

                % generation
                   [offspring, calls,r,m,flag,c_flag,archive] = OperatorCMFEA.generate(1, population, Tasks, rmp(generation-1), obj.mu, obj.mum, obj.probswap, mu_tasks, Sigma_tasks,r,m,flag,c_flag,archive,pop_size);
                fnceval_calls = fnceval_calls + calls;

                % selection
    %            [population, bestobj, bestCV, data.bestX, feasible_rate] = selectMF_FP2(population, offspring, Tasks, pop_size, bestobj, bestCV, data.bestX,archive);
                [population, bestobj, bestCV, bestX] = selectCMFEA(population, offspring, Tasks, pop_size, bestobj, bestCV, bestX,archive);
                for t=1:length(Tasks)
                        [~,idx] = find([population.skill_factor] == t);
                        i=randi(length(idx));
                        x_c=population(idx(i));
                        k=randi(Tasks.dims);
                        for i = 1:length(population)
                             constraint_violation(i) = population(i).constraint_violation(t);
                        end 
                        max_cv(t)=max(constraint_violation);
                        x = (constraint_violation == max_cv(t));
                        idx = 1:length(x);
                        idx = idx(x);
                        i=randi(length(idx));
                        x_e=population(idx(i));
                        rand_num=rand();                      
                        x_c.rnvec(k)=rand_num;
                        if x_c.factorial_costs(t) < x_e.factorial_costs(t)
                            population(idx(i))=x_c;
                        end
%                     end
                end
                 
               % update rmp
               if r==0
                   prob_1=0;
               else
                   prob_1=flag/r;
               end
               if m==0
                   prob_2=0;
               else
                   prob_2=c_flag/m;
               end
                              
               if generation >max_generation*obj.ro
                    if prob_1 <prob_2
                        rmp(generation)=max(rmp(generation-1)-obj.c^2,0);                        
                    end
                    if prob_1 >prob_2
                        rmp(generation)=min(rmp(generation-1)+obj.c^2,1);
                    end
                end
                
                rmp(generation)
                
                convergence(:, generation) = bestobj;
                convergence_cv(:, generation) = bestCV;

                % Updates of the progresisonal representation models
                [mu_tasks, Sigma_tasks] = DistributionUpdate(mu_tasks, Sigma_tasks, population, length(Tasks));
            end
            data.convergence = gen2eva(convergence);
            data.convergence_cv = gen2eva(convergence_cv);
            data.bestX = uni2real(bestX, Tasks);
        end
    end
end
