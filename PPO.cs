using Tensorflow;
using static Tensorflow.Binding;
using NumSharp;
using System;
using System.Collections.Generic;

namespace ppo.net
{
    // Implementação da classe ppo //
    class _PPO
    {
        /// <summary>
        /// Classe PPO agrega:
        ///  As redes neurais ATOR e CRITICA;
        ///  Função para atualizar as redes neurais;
        ///  Função para obter o valor de aprendizagem da CRITICA;
        ///  Função para treinar as redes neurais;
        ///  Função para escolher uma ação;
        /// </summary>


        #region Configurações
        float ACTOR_LEARNING_RATE = (float)0.0001;  // Taxa de aprendizado do ATOR
        float CRITIC_LEARNING_RATE = (float)0.0002; // Taxa de aprendizado da CRITICA
        int ACTOR_UPDATE_STEPS = 20;                // Quantidade de vezes que o treinamento do ATOR vai tomar action cadeia de dados de batch
        int CRITIC_UPDATE_STEPS = 20;               // Quantidade de vezes que o treinamento da CRITICA vai tomar action cadeia de dados de batch
        int STATES = 3;                             // STATES é action dimensão do estado, ou seja, quantas entradas ele terá
        int ACTIONS = 1;                            // ACTIONS é action dimensão das ações, ou seja, quantas ações podem ser executadas
        #endregion

        #region Variables
        private Session session;
        // Placeholders
        private Tensor state_placeholder;
        private Tensor action_placeholder;
        private Tensor advantage_placeholder;
        private Tensor discounted_reward_placeholder;
        // Layers
        //  Critic
        private Tensor critic_1_layer;
        private Tensor critic_value_layer;
        private Operation critic_optmized_trainer;
        private Tensor critic_loss;
        //  Actor
        private Normal actor_policy;
        private List<Tensor> actor_policy_parameters;
        private Normal actor_old_policy;
        private List<Tensor> actor_old_policy_parameters;
        private Operation actor_old_policy_trainer;
        private Tensor actor_loss;
        //
        private Tensor ADVANTAGE;
        private Tensor sample_actor_old_policy;
        private Tensor update_actor_old_policy;
        private Tensor ratio;
        private Tensor surrogate;
        
        private NDArray advantage;
        #endregion

        Dictionary<string, dynamic> METHOD = new Dictionary<string, dynamic>()
        {
            {"name", "clip"},   // Método de clip (Clipped surrogate objective) sugerido pelos papéis como mais eficiente
            {"epsilon", 0.2}    // epsilon=0.2 Valor de epsilon sugerido pelos papéis
        };

        _PPO()   // Construtor da Classe
        {
            this.session = tf.Session();   //inicializar uma seção do TensorFlow

            // Declaração das entradas das redes:
            this.state_placeholder = tf.placeholder(    // Estado do ambiente: action rede recebe o estado do ambiente através desse placeholder
                tf.float32,                             //   Tipo do placeholder
                shape: (-1, STATES),                    //   Dimensões do placeholder
                name: "state_placeholder"               //   Nome do placeholder
            );

            this.action_placeholder = tf.placeholder(   // Ação escolhida pela rede é informada através desse placeholder
                tf.float32,                             //   Tipo do placeholder
                shape: (-1, ACTIONS),                   //   Dimensões do placeholder
                name: "action_placeholder"              //   Nome do placeholder
            );

            this.advantage_placeholder = tf.placeholder(    // Calculo do ganho que action rede obteve no episódio, calculado fora da classe PPO.
                tf.float32,                                 //   Tipo do placeholder
                shape: (-1, 1),                             //   Tamanho do placeholder
                name: "advantage_placeholder"               //   Nome do placeholder
            );                                              // Esse placeholder é usado para treinar tanto o ATOR quanto action CRITICA

            //      // CRITICA:
            using (tf.variable_scope("critic"))
            {
                // Criação da rede neural:
                critic_1_layer = tf.layers.dense(   // Camada 1 entrada da Critica:
                    inputs: this.state_placeholder, //   this.state_placeholder é o placeholder do estado, funciona como entrada da rede
                    100,                            //   100 é o numero de neurônios
                    activation: tf.nn.relu(),       //   Relu é o tipo de ativação da saída da camada
                    name: "critic_1_layer"          //   name é o nome da camada
                );

                this.critic_value_layer = tf.layers.dense(  // Camada de saída de valores da CRITICA:
                    inputs: critic_1_layer,                 //   critic_1_layer é action variável referente action primeira camada da rede,
                    1,                                      //   1 é action quantidade de saídas da rede
                    name: "critic_value_layer"              //   name é o nome da camada
                );                                          //   A saída dessa rede será o Q-Value, o status do progresso do aprendizado
            }
            // Método de treinamento para o CRITICA, ou seja, o método de aprendizagem:
            using (tf.variable_scope("critic_trainer"))
            {
                this.discounted_reward_placeholder = tf.placeholder(    // A recompensa de cada episódio é inserida na rede através desse placeholder
                    tf.float32,                                         //   Tipo do placeholder
                    shape: (-1, 1),                                     //   Dimensões do placeholder
                    name: "discounted_reward_placeholder"               //   Nome do placeholder
                );

                this.ADVANTAGE = this.discounted_reward_placeholder - this.critic_value_layer;   // Através da recompensa discounted_reward_placeholder subtraída pelo valor de aprendizagem critic_value_layer obtemos action vantagem

                this.critic_loss = tf.reduce_mean(  // tf.reduce_mean calcula action média.
                    tf.square(                      // tf.square calcula o quadrado
                        this.ADVANTAGE              // da vantagem
                    )
                );                                  // !  Aqui obtemos em critic_loss o Loss, ou seja, action Perda da CRITICA

                this.critic_optmized_trainer = tf.train.AdamOptimizer(CRITIC_LEARNING_RATE).minimize(this.critic_loss); // Utilizamos o otimizador ADAM, com action taxa de aprendizado da CRITICA CRITIC_LEARNING_RATE com action função minimize processamos os gradientes da CRITICA através da perda da CRITICA em critic_loss. (outra alternativa é o otimizador SGD)
            }
            // ATOR:
            //   Politica atual
            (actor_policy, actor_policy_parameters) = this.build_actor_network(name: "actor_policy", trainable: true);  // Criação da rede neural actor_policy para action politica atual do ATOR através da função build_anet, definindo como treinável actor_policy é action saída da rede e actor_policy_parameters são os pesos (estado atual) da rede Os pesos actor_policy_parameters são utilizados para atualizar as politicas atual action antiga.

            using (tf.variable_scope("sample_action"))
            {
                this.sample_actor_old_policy = tf.squeeze(actor_policy.sample(1), axis: 0); // Tira uma amostra de ação da politica atual actor_policy do ATOR // !
            }
            //   Politica antiga
            (actor_old_policy, actor_old_policy_parameters) = this.build_actor_network("actor_old_policy", trainable: false);   // Criação da rede neural actor_old_policy para action politica antiga do ATOR através da função build_anet, definindo como não treinável

            using (tf.variable_scope("update_oldpi"))   // Atualização dos pesos dos pesos de actor_old_policy tendo como referencia os pesos de actor_policy
            {
                
                foreach (var (politic, old_politic) in zip(actor_policy_parameters, actor_old_policy_parameters))   // A cada atualização da rede, os pesos da politica atual passam para action politica antiga
                {
                    this.update_actor_old_policy += old_politic.assign(politic);
                }
                 
                    // Update_oldpi_op acumula todos os valores de actor_policy ao decorrer do episodio
            }

            // Implementação da função de perda PPO
            using (tf.variable_scope("loss"))   // Função de perda:
            {
                using (tf.variable_scope("surrogate_pp"))
                {
                    ratio = actor_policy.prob(this.action_placeholder) / actor_old_policy.prob(this.action_placeholder); // !
                    // O Ratio é action razão da probabilidade da ação actor_policy na politica nova  pela probabilidade da ação action_placeholder na politica antiga.
                    surrogate = ratio * this.advantage_placeholder; // Surrogate é action Razão multiplicada pela vantagem

                    this.actor_loss = -tf.reduce_mean(          // tf.educe_mean calcula action negativa da média do
                        tf.minimum(                             //   menor valor entre
                            surrogate,                          //       o Surrogate e
                            this.advantage_placeholder *        //       action multiplicação da vantagem
                                tf.clip_by_value(               //           pelo ratio clipado (limitado) por
                                    ratio,                      //
                                    1.0 - METHOD["epsilon"],    //                no máximo 1 - o método Clipped surrogate objective
                                    1.0 + METHOD["epsilon"]     //                no minimo 1 + o método Clipped surrogate objective
                                )                               //
                        )                                       // Obtendo assim em actor_loss action perda do Ator
                    );
                }
            }
            // Método de treinamento para o ATOR, ou seja, o método de aprendizagem:
            using (tf.variable_scope("actor_trainer"))
            {
                this.actor_old_policy_trainer = tf.train.AdamOptimizer(ACTOR_LEARNING_RATE).minimize(this.actor_loss);  // Utilizamos o otimizador ADAM, com action taxa de aprendizado do ATOR ACTOR_LEARNING_RATE e com minimize processamos os gradientes do ATOR através da perda do ATOR em actor_loss
            }

            tf.summary.FileWriter("log/", this.session.graph);      // Salvando o modelo na pasta log para analise futura no tensorboard

            this.session.run(tf.global_variables_initializer());    // Inicializando todas as variáveis definidas
        }

        // Função de atualização
        void update_network(NDArray state, NDArray action, NDArray reward)  // Recebe o estado, action ação e action recompensa // !
        {
            this.session.run(this.update_actor_old_policy);                 // Executa action matriz update_actor_old_policy que contem todos os pesos de actor_policy/actor_old_policy

            // Atualiza o ATOR
            advantage = this.session.run(
                this.ADVANTAGE,                                             // Calcula action vantagem, ou seja, action recompensa do ATOR
                new FeedItem(this.state_placeholder, state),                // Recebe o estado
                new FeedItem(this.discounted_reward_placeholder, reward)    // Recebe action recompensa
            );

            for (int step = 0; step <= ACTOR_UPDATE_STEPS; step++)          // ACTOR_UPDATE_STEPS é quantas vezes que action rede vai ser atualizada
            {
                this.session.run(
                    this.actor_old_policy_trainer,                          // Treina o ator
                    new FeedItem(this.state_placeholder, state),            // Recebe o estado
                    new FeedItem(this.action_placeholder, action),          // Recebe action ação
                    new FeedItem(this.advantage_placeholder, advantage)     // Recebe o avanço
                );
            }
            // Atualiza action CRITICA através da função de treinamento
            for (int step = 0; step <= CRITIC_UPDATE_STEPS; step++)             // CRITIC_UPDATE_STEPS é quantas vezes que action rede vai ser atualizada
            {
                this.session.run(                                               // Executa o treinamento da critica
                    this.critic_optmized_trainer,                               //   critic_optmized_trainer é o treinamento da critica
                    new FeedItem(this.state_placeholder, state),                //   state_placeholder é o placeholder que recebe estado state do ambiente
                    new FeedItem(this.discounted_reward_placeholder, reward)    //   discounted_reward_placeholder é o placeholder que recebe action recompensa reward do ambiente
                );
            }
        }

        private (Normal normal_distribuition, List<Tensor> parameters) build_actor_network(string name, bool trainable) // !
        {
            // Constrói as redes neurais do ATOR
            //    name é o nome da rede
            //    trainable determina se action rede é treinável ou não
            using (tf.variable_scope(name))
            {
                Tensor actor_1_layer = tf.layers.dense(   // Camada 1 entrada do ATOR:
                    inputs: this.state_placeholder, //   this.state_placeholder é o placeholder do estado, funciona como entrada pra rede
                    100,                            //   100 é o numero de neurônios
                    tf.nn.relu(),                   //   Relu é o tipo de ativação da saída da rede
                    trainable: trainable            //   trainable determina se action rede é treinável ou não
                );

                //   Calcula action ação que vai ser tomada
                Tensor mu = 2 * tf.layers.dense(    // Camada mu do ATOR
                    inputs: actor_1_layer,         //   actor_1_layer é action entrada da camada
                    ACTIONS,                        //   ACTIONS
                    tf.nn.tanh(),                   //   tanh é o tipo de ativação da saída da camada, retorna um valor entre 1 e -1
                    trainable: trainable,           //   trainable determina se action rede é treinável ou não
                    name: "mu_" + name              //   name é o nome da camada
                );                                  //   O resultado é multiplicado por 2 para se adequar ao ambiente, que trabalha com um range 2 e -2.


                //   Calcula o desvio padrão, o range onde estará action possibilidade de ação
                Tensor sigma = tf.layers.dense(     // Camada sigma do ATOR
                    inputs: actor_1_layer,         //   actor_1_layer é action entrada da camada
                    ACTIONS,                        //   ACTIONS
                    activation: tf.nn.relu(),   //   softplus é o tipo de ativação da saída da camada // !
                    trainable: trainable,           //   trainable determina se action rede é treinável ou não
                    name: "sigma_" + name           //   name é o nome da camada
                );

                //Tensor normal_distribuition = tf.distributions.Tensor(  // Transforma em tensor action saída mu da rede, considerando sigma // !
                Normal normal_distribuition = tf.distributions.Normal(  // Transforma em tensor action saída mu da rede, considerando sigma // !
                    loc: mu,                                            // Loc é action média
                    scale: sigma
                );
                
                // Coleta em parameters os pesos das camadas actor_1_layer, mu/2 e sigma do escopo atual
                List<Tensor> parameters = tf.get_collection<Tensor>(key: tf.GraphKeys.GLOBAL_VARIABLES, scope: name);
                return (normal_distribuition, parameters);    // Retorna action ação e os pesos atuais das redes para serem armazenados na
            }
        }

        public NDArray choose_action(NDArray state)         // Recebe o estado state e retorna uma ação action
        {
            state = state["newaxis, :"];                    //   Recebe o estado state e
            NDArray a = this.session.run(                   //   Executa sample_actor_old_policy
                this.sample_actor_old_policy,
                new FeedItem(this.state_placeholder, state) //   com o placeholder state_placeholder que recebe o estado state e armazena action ação em action
            )[0];                                           //  seleciona action posição zero do arrai resultante
            return (np.clip(a, -2, 2));                     //   Retorna um valor de ação action clipado entre -2 e 2
        }

        public NDArray get_value(NDArray state)             // Recebe o estado state e retorna o valor da taxa de aprendizagem da CRITICA
        {
            if (state.ndim < 2)
            {
                state = state["newaxis, :"];
            }
            return this.session.run(                        // Retorna action taxa de aprendizagem da CRITICA
                this.critic_value_layer,                    // critic_value_layer é action saída de valores da CRITICA
                new FeedItem(this.state_placeholder, state) // state_placeholder é o placeholder que recebe o estado state
            )[0, 0];                                        //  seleciona action posição zero, zero do arrai resultante
        }

    }
}