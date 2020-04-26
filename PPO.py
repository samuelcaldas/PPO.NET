# Uma simples versão de PPO (Proximal Policy Optimization) baseada em:
# 1.  [https://arxiv.org/abs/1707.02286]
# 2.  [https://arxiv.org/abs/1707.06347]
# Veja mais nesses tutoriais: https://morvanzhou.github.io/tutorials
# e nesse vídeo: https://www.youtube.com/watch?v=lehLSoMPmcM&t=144s

#   Importações #
import tensorflow.compat.v1 as tf   # Workaround para retrocompatibilidade
tf.disable_v2_behavior()            # com tensorflow v1
import numpy as np                  # Numpy para trabalhar com arrays

#   Configurações #
GAMMA = 0.9                 # Avanço (?)
ACTOR_LEARNING_RATE = 0.0001    # Taxa de aprendizado do ATOR
CRITIC_LEARNING_RATE = 0.0002   # Taxa de aprendizado da CRITICA
ACTOR_UPDATE_STEPS = 20         # Quantidade de vezes que o treinamento do ATOR vai tomar a cadeia de dados de batch
CRITIC_UPDATE_STEPS = 20        # Quantidade de vezes que o treinamento da CRITICA vai tomar a cadeia de dados de batch
STATES = 3                   # S_DIM é a dimensão do estado, ou seja, quantas entradas ele terá
ACTIONS = 1                  # A_DIM é a dimensão das ações, ou seja, quantas ações podem ser executadas
METHOD = dict(name='clip',  # Método de clip (Clipped surrogate objective) sugerido pelos papéis como
                            # mais eficiente
    epsilon=0.2             # epsilon=0.2 Valor de epsilon sugerido pelos papéis
)
                                            

#   Implementação da classe ppo #
class PPO(object):  
    # Classe PPO agrega:
    #   As redes neurais ATOR e CRITICA;
    #   Função para atualizar as redes neurais;
    #   Função para obter o valor de aprendizagem da CRITICA;
    #   Função para treinar as redes neurais;
    #   Função para escolher uma ação;

    def __init__(self): # Construtor da Classe
        self.session = tf.Session()    #inicializar uma seção do TensorFlow
        # Declaração das entradas das redes:
        self.state_placeholder = tf.placeholder(# Estado do ambiente: a rede recebe o estado do ambiente através desse placeholder
            tf.float32,             #   Tipo do placeholder
            [None, STATES],      #   Dimensões do placeholder
            'state_placeholder'                 #   Nome do placeholder
        )  

        self.action_placeholder = tf.placeholder(   # Ação escolhida pela rede é informada através desse placeholder
            tf.float32,                 #   Tipo do placeholder
            [None, ACTIONS],         #   Dimensões do placeholder
            'action_placeholder'                    #   Nome do placeholder
        )  

        self.advantage_placeholder = tf.placeholder(    # Calculo do ganho que a rede obteve no episódio, calculado fora da classe PPO.
            tf.float32,                 #   Tipo do placeholder
            [None, 1],                  #   Tamanho do placeholder
            'advantage_placeholder'                 #   Nome do placeholder
        )                               # Esse placeholder é usado para treinar tanto o
                                        # ATOR quanto a CRITICA

        # CRITICA:
        with tf.variable_scope('critic'):   
            # Criação da rede neural:
            critic_1_layer = tf.layers.dense(       # Camada 1 entrada da Critica:
                self.state_placeholder,             #   self.tfs é o placeholder do estado, funciona como entrada da
                                        #   rede
                100,                    #   100 é o numero de neurônios
                tf.nn.relu,             #   Relu é o tipo de ativação da saída da camada
                name='critic_1_layer'    #   name é o nome da camada
            )   

            self.critic_value_layer = tf.layers.dense(   # Camada de saída de valores da CRITICA:
                critic_1_layer,                     #   l1 é a variável referente a primeira camada da rede,
                1,                      #   1 é a quantidade de saídas da rede
                name = 'critic_value_layer'        #   name é o nome da camada
            )                           #   A saída dessa rede será o Q-Value, o status do
                                        #   progresso do aprendizado

        # Método de treinamento para o CRITICA, ou seja, o método de
        # aprendizagem:
        with tf.variable_scope('critic_trainer'):
            self.discounted_reward_placeholder = tf.placeholder(    # A recompensa de cada episódio é inserida na rede através desse placeholder
                tf.float32,                             #   Tipo do placeholder
                [None, 1],                              #   Dimensões do placeholder
                'discounted_reward_placeholder'                          #   Nome do placeholder
            )     
            self.advantage_placeholder = self.discounted_reward_placeholder - self.critic_value_layer  # Através da recompensa discounted_r/tfdc_r subtraída pelo valor de aprendizagem V_layer/v obtemos a vantagem
            self.critic_loss = tf.reduce_mean(    # tf.reduce_mean calcula a média.
                tf.square(                  # tf.square calcula o quadrado
                    self.advantage_placeholder          # da vantagem
                ))                          # !  Através disso obtemos em closs o Loss ou a
                                            # Perda da CRITICA
                                               
            self.critic_optmized_trainer = tf.train.AdamOptimizer(CRITIC_LEARNING_RATE).minimize(self.critic_loss)  # Utilizamos o otimizador ADAM, com a taxa de aprendizado da CRITICA C_LR
                                                                                                # com a função minimize processamos os gradientes da CRITICA através da perda da CRITICA em closs
                                                                                                #   Poderíamos usar também o SGD como otimizador.

        # ATOR:
        #   Politica atual
        actor_policy, actor_policy_parameters = self.build_actor_network('actor_policy', trainable=True)  # Criação da rede neural (pi) para a politica atual do ATOR
                                                                        # através da função build_anet, definindo como treinável
                                                                        # pi é a saída da rede e pi_params são os pesos (estado atual) da rede
                                                                        # Os pesos pi_params são utilizados para atualizar as politicas atual a antiga.

        with tf.variable_scope('sample_action'):
            self.sample_actor_old_policy = tf.squeeze(actor_policy.sample(1), axis=0)       # Tira uma amostra de ação da politica atual pi do ATOR

        #   Politica antiga
        actor_old_policy, actor_old_policy_parameters = self.build_actor_network('actor_old_policy', trainable=False)    # Criação da rede neural oldpi para a politica antiga do ATOR através da função build_anet, definindo como não treinável

        with tf.variable_scope('update_actor_old_policy'):                                                 # Atualização dos pesos dos pesos de oldpi tendo como referencia os pesos de pi
            self.update_actor_old_policy = [old_policy.assign(policy) for policy, old_policy in zip(actor_policy_parameters, actor_old_policy_parameters)] # A cada atualização da rede, os pesos da politica atual passam para a politica antiga Update_oldpi_op acumula todos os valores de pi ao decorrer do episodio

        # Implementação da função de perda PPO
        with tf.variable_scope('loss'): # Função de perda:
            with tf.variable_scope('surrogate_pp'):
                ratio = actor_policy.prob(self.action_placeholder) / actor_old_policy.prob(self.action_placeholder)    # O Ratio é a razão da probabilidade da ação tfa na politica nova pela probabilidade da ação tfa na politica antiga.
                surrogate = ratio * self.advantage_placeholder                           # Surrogate é a Razão multiplicada pela vantagem

            self.actor_loss = -tf.reduce_mean(               # tf.educe_mean calcula a negativa da média do
                tf.minimum(                             #   menor valor entre
                    surrogate,                               #       o Surrogate e
                    self.advantage_placeholder *                        #       a multiplicação da vantagem
                        tf.clip_by_value(               #           pelo ratio clipado (limitado) por
                            ratio,                      #
                            1. - METHOD['epsilon'],     #                no máximo 1 - o método Clipped surrogate
                                                        #                objective
                            1. + METHOD['epsilon']      #                no minimo 1 + o método Clipped surrogate
                                                        #                objective
                        )                               #
                )                                       # Obtendo assim em aloss a
                                                        # perda do Ator
            )

        # Método de treinamento para o ATOR, ou seja, o método de aprendizagem:
        with tf.variable_scope('actor_trainer'):
            self.actor_old_policy_trainer = tf.train.AdamOptimizer(ACTOR_LEARNING_RATE).minimize(self.actor_loss)  # Utilizamos o otimizador ADAM, com a taxa de aprendizado do ATOR A_LR
                                                                                # com minimize processamos os gradientes do ATOR através da perda do ATOR em aloss

        tf.summary.FileWriter("log/", self.session.graph)      # Salvando o modelo na pasta log para analise futura no tensorboard

        self.session.run(tf.global_variables_initializer())    # Inicializando todas as variáveis definidas
    
    # Função de atualização
    def update_network(self, state, action, reward):              # Recebe o estado, a ação e a recompensa
        self.session.run(self.update_actor_old_policy) # Executa a matriz update_oldpi_op que contem todos os pesos de pi/oldpi
        
        # Atualiza o ATOR
        advantage = self.session.run(
            self.advantage_placeholder,     # Calcula a vantagem, ou seja, a recompensa do ATOR
            {
                self.state_placeholder: state,    # Recebe o estado
                self.discounted_reward_placeholder: reward  # Recebe a recompensa
            })
        [self.session.run(
            self.actor_old_policy_trainer,     # Treina o ator
            {
                self.state_placeholder: state,    # Recebe o estado
                self.action_placeholder: action,    # Recebe a ação
                self.advantage_placeholder: advantage # Recebe o avanço
            }) for _ in range(ACTOR_UPDATE_STEPS)]   # A_UPDATE_STEPS é quantas vezes que a rede vai ser atualizada

        # Atualiza a CRITICA através da função de treinamento
        [self.session.run(         # Executa o treinamento da critica
            self.critic_optmized_trainer,     #   ctrain_op é o treinamento da critica
            {
                self.state_placeholder: state,                      #   tfs é o placehoder que recebe estado s do ambiente
                self.discounted_reward_placeholder: reward           #   tfdc_r é o placeholder que recebe a recompensa r do ambiente
            }) for _ in range(CRITIC_UPDATE_STEPS)] # C_UPDATE_STEPS é quantas vezes que a rede vai ser atualizada
                                                                                                        
                                                                                                        
                                                                                                        
                                                                                                        

    def build_actor_network(self, name, trainable): 
        # Constrói as redes neurais do ATOR
        #    name é o nome da rede
        #    trainable determina se a rede é treinável ou não
        with tf.variable_scope(name):   
            actor_1_layer = tf.layers.dense(       # Camada 1 entrada do ATOR:
                self.state_placeholder,               #   self.tfs é o placeholder do estado, funciona como entrada
                                        #   pra rede
                100,                    #   100 é o numero de neurônios
                tf.nn.relu,             #   Relu é o tipo de ativação da saída da rede
                trainable=trainable     #   trainable determina se a rede é treinável ou não
            )

            #   Calcula a ação que vai ser tomada
            mu = 2 * tf.layers.dense(   # Camada mu do ATOR
                actor_1_layer,                     #   l1 é a entrada da camada
                ACTIONS,                  #   A_DIM
                tf.nn.tanh,             #   tanh é o tipo de ativação da saída da camada, retorna um valor entre 1 e -1
                trainable=trainable,    #   trainable determina se a rede é treinável ou não
                name = 'mu_' + name     #   name é o nome da camada
            )                           #   O resultado é multiplicado por 2 para se adequar
                                        #   ao ambiente, que trabalha com um range 2 e -2.

            
            #   Calcula o desvio padrão, o range onde estará a possibilidade de
            #   ação
            sigma = tf.layers.dense(    # Camada sigma do ATOR
                actor_1_layer,                     #   l1 é a entrada da camada
                ACTIONS,                  #   A_DIM
                tf.nn.softplus,         #   softplus é o tipo de ativação da saída da camada
                trainable=trainable,    #   trainable determina se a rede é treinável ou não
                name ='sigma_' + name   #   name é o nome da camada
            )    

            normal_distribuition = tf.distributions.Normal(# Normaliza a saída mu da rede, considerando sigma
                loc=mu,                             # Loc é a média
                scale=sigma)            
                                                                                
        parameters = tf.get_collection(             # Coleta em params os pesos
            tf.GraphKeys.GLOBAL_VARIABLES,      # das camadas l1,mu/2 e sigma
            scope=name                          # do escopo atual
        )   
        return normal_distribuition, parameters    # Retorna a ação e os pesos atuais das redes para serem armazenados na politica antiga.

    def choose_action(self, state): # Recebe o estado s e retorna uma ação a
        state = state[np.newaxis, :]    #   Recebe o estado s e
        action = self.session.run(self.sample_actor_old_policy,   #   Executa sample_op
            {self.state_placeholder: state}     #   com o placeholder tfs que recebe o estado s e armazena a ação em a
        )[0]    #
        return np.clip(action, -2, 2)    #   Retorna um valor de ação a clipado entre -2 e 2

    def get_value(self, state):             # Recebe o estado s e retorna o valor da taxa de aprendizagem da
                                    # CRITICA
        if state.ndim < 2: state = state[np.newaxis, :] #
        return self.session.run(   # Retorna a taxa de aprendizagem da CRITICA
            self.critic_value_layer,             # v é a saída de valores da CRITICA
            {self.state_placeholder: state}       # tfs é o placeholder que recebe o estado s
        )[0, 0] #