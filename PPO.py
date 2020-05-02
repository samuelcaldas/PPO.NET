# Uma simples versão de PPO (Proximal Policy Optimization) baseada em:
# 1. [https://arxiv.org/abs/1707.02286]
# 2. [https://arxiv.org/abs/1707.06347]
# Veja mais nesses tutoriais: https://morvanzhou.github.io/tutorials
# e nesse video: https://www.youtube.com/watch?v=lehLSoMPmcM&t=144s

#   Importações   #

import tensorflow.compat.v1 as tf   # Workaround para retrocompatibilidade 
tf.disable_v2_behavior()            # com tensorflow v1
import numpy as np                  # Numpy para trabalhar com arrays

#   Configurações   #

GAMMA = 0.9             # Avanço (?)
A_LR = 0.0001           # Taxa de aprendizado do ATOR
C_LR = 0.0002           # Taxa de aprendizado da CRITICA
A_UPDATE_STEPS = 20     # Quantidade de vezes que o treinamento do ATOR vai tomar a cadeia de dados de batch
C_UPDATE_STEPS = 20     # Quantidade de vezes que o treinamento da CRITICA vai tomar a cadeia de dados de batch
S_DIM = 3               # S_DIM é a dimensão do estado, ou seja, quantas entradas ele terá
A_DIM = 1               # A_DIM é a dimensão das ações, ou seja, quantas ações podem ser executadas

METHOD = dict(
    name='clip',    # Método de clip (Clipped surrogate objective) sugerido pelos papéis como mais eficiente
    epsilon=0.2     # epsilon=0.2 Valor de epsilon sugerido pelos papéis
)
                                            

#   Implementação da classe ppo   #

class PPO(object):  
    # Classe PPO agrega:
    #   As redes neurais ATOR e CRITICA;
    #   Função para atualizar as redes neurais;
    #   Função para obter o valor de aprendizagem da CRITICA;
    #   Função para treinar as redes neurais;
    #   Função para escolher uma ação;

    def __init__(self): # Construtor da Classe
        self.sess = tf.Session()    #inicializar uma seção do TensorFlow
        # Declaração das entradas das redes:
        self.tfs = tf.placeholder(  # Estado do ambiente: a rede recebe o estado do ambiente através desse placeholder
            tf.float32,             #   Tipo do placeholder
            [None, S_DIM ],         #   Dimensões do placeholder
            'state'                 #   Nome do placeholder
        )  

        self.tfa = tf.placeholder(  # Ação escolhida pela rede é informada através desse placeholder 
            tf.float32,                 #   Tipo do placeholder
            [None, A_DIM],              #   Dimensões do placeholder
            'action'                    #   Nome do placeholder
        )  

        self.tfadv = tf.placeholder(    # Calculo do ganho que a rede obteve no episódio, calculado fora da classe PPO.
            tf.float32,                 #   Tipo do placeholder
            [None, 1],                  #   Tamanho do placeholder
            'advantage'                 #   Nome do placeholder
        )                               # Esse placeholder é usado para treinar tanto o ATOR quanto a CRITICA

        # CRITICA:
        with tf.variable_scope('critic'):   
            # Criação da rede neural:
            l1 = tf.layers.dense(       # Camada 1 entrada da Critica: 
                self.tfs,               #   self.tfs é o placeholder do estado, funciona como entrada da rede
                100,                    #   100 é o numero de neurônios 
                tf.nn.relu,             #   Relu é o tipo de ativação da saída da camada
                name='layer1-critic'    #   name é o nome da camada
            )   

            self.v = tf.layers.dense(   # Camada de saída de valores da CRITICA: 
                l1,                     #   l1 é a variável referente a primeira camada da rede, 
                1,                      #   1 é a quantidade de saídas da rede
                name = 'V_layer'        #   name é o nome da camada  
            )                           #   A saída dessa rede será o Q-Value, o status do progresso do aprendizado

        # Método de treinamento para o CRITICA, ou seja, o método de aprendizagem:
        with tf.variable_scope('ctrain'):
            self.tfdc_r = tf.placeholder(   # A recompensa de cada episódio é inserida na rede através desse placeholder
                tf.float32,                 #   Tipo do placeholder
                [None, 1],                  #   Dimensões do placeholder
                'discounted_r'              #   Nome do placeholder
            )     
            self.advantage = self.tfdc_r - self.v   # Através da recompensa discounted_r/tfdc_r subtraída pelo
                                                    # valor de aprendizagem V_layer/v obtemos a vantagem
            self.closs = tf.reduce_mean(    # tf.reduce_mean calcula a média. 
                tf.square(                  # tf.square calcula o quadrado
                    self.advantage          # da vantagem
                )
            )                               # ! Através disso obtemos em closs o Loss ou a Perda da CRITICA                  
                                               
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)  # Utilizamos o otimizador ADAM, com a taxa de aprendizado da CRITICA C_LR
                                                                                # com a função minimize processamos os gradientes da CRITICA através da perda da CRITICA em closs
                                                                                #   Poderíamos usar também o SGD como otimizador.

        # ATOR:
        #   Politica atual
        pi, pi_params = self._build_anet('pi', trainable=True)                  # Criação da rede neural (pi) para a politica atual do ATOR através da função build_anet, definindo como treinável
                                                                                #   pi é a saída da rede e pi_params são os pesos (estado atual) da rede
                                                                                #   Os pesos pi_params são utilizados para atualizar as politicas atual a antiga.

        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)               # Tira uma amostra de ação da politica atual pi do ATOR

        #   Politica antiga
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)    # Criação da rede neural oldpi para a politica antiga do ATOR através da função build_anet, definindo como não treinável

        with tf.variable_scope('update_oldpi'):                                                 # Atualização dos pesos dos pesos de oldpi tendo como referencia os pesos de pi
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)] # A cada atualização da rede, os pesos da politica atual passam para a politica antiga
                                                                                                # Update_oldpi_op acumula todos os valores de pi ao decorrer do episodio

        # Implementação da função de perda PPO
        with tf.variable_scope('loss'): # Função de perda:
            with tf.variable_scope('surrogate_pp'):
                ratio = pi.prob(self.tfa) / oldpi.prob(self.tfa)    # O Ratio é a razão da probabilidade da ação tfa na politica nova 
                                                                    # pela probabilidade da ação tfa na politica antiga.
                surr = ratio * self.tfadv                           # Surrogate é a Razão multiplicada pela vantagem

            self.aloss = -tf.reduce_mean(                           # tf.educe_mean calcula a negativa da média do
                tf.minimum(                                         #   menor valor entre
                    surr,                                           #       o Surrogate e
                    self.tfadv*                                     #       a multiplicação da vantagem
                        tf.clip_by_value(                           #           pelo ratio clipado (limitado) por
                            ratio,                                  #               
                            1.-METHOD['epsilon'],                   #                no máximo 1 - o método Clipped surrogate objective
                            1.+METHOD['epsilon']                    #                no minimo 1 + o método Clipped surrogate objective              
                        )                                           # 
                )                                                   # Obtendo assim em aloss a perda do Ator
            )

        # Método de treinamento para o ATOR, ou seja, o método de aprendizagem:
        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)  # Utilizamos o otimizador ADAM, com a taxa de aprendizado do ATOR A_LR
                                                                                # com minimize processamos os gradientes do ATOR através da perda do ATOR em aloss

        tf.summary.FileWriter("log/", self.sess.graph)      # Salvando o modelo na pasta log para analise futura no tensorboard

        self.sess.run(tf.global_variables_initializer())    # Inicializando todas as vária veis definidas
    
    # Função de atualização
    def update_network(self, s, a, r):      # Recebe o estado, a ação e a recompensa
        self.sess.run(self.update_oldpi_op) # Executa a matriz update_oldpi_op que contem todos os pesos de pi/oldpi
        
        # Atualiza o ATOR
        adv = self.sess.run(
            self.advantage,     # Calcula a vantagem, ou seja, a recompensa do ATOR 
            {
                self.tfs: s,    # Recebe o estado
                self.tfdc_r: r  # Recebe a recompensa
            }
        )
        [self.sess.run(
            self.atrain_op,     # Treina o ator
            {
                self.tfs: s,    # Recebe o estado
                self.tfa: a,    # Recebe a ação
                self.tfadv: adv # Recebe o avanço
            }
        ) for _ in range(A_UPDATE_STEPS)]   # A_UPDATE_STEPS é quantas vezes que a rede vai ser atualizada

        # Atualiza a CRITICA através da função de treinamento
        [self.sess.run(         # Executa o treinamento da critica
            self.ctrain_op,     #   ctrain_op é o treinamento da critica
            {
                self.tfs: s,    #   tfs é o placehoder que recebe estado s do ambiente
                self.tfdc_r: r  #   tfdc_r é o placeholder que recebe a recompensa r do ambiente
            }
        ) for _ in range(C_UPDATE_STEPS)]   # C_UPDATE_STEPS é quantas vezes que a rede vai ser atualizada
                                                                                                        
                                                                                                        
                                                                                                        
                                                                                                        

    def _build_anet(self, name, trainable): 
        # Constrói as redes neurais do ATOR
        #    name é o nome da rede
        #    trainable determina se a rede é treinável ou não
        with tf.variable_scope(name):   
            l1 = tf.layers.dense(       # Camada 1 entrada do ATOR: 
                self.tfs,               #   self.tfs é o placeholder do estado, funciona como entrada pra rede
                100,                    #   100 é o numero de neurônios 
                tf.nn.relu,             #   Relu é o tipo de ativação da saída da rede
                trainable=trainable     #   trainable determina se a rede é treinável ou não
            )

            #   Calcula a ação que vai ser tomada
            mu = 2 * tf.layers.dense(    # Camada mu do ATOR  
                l1,                     #   l1 é a entrada da camada
                A_DIM,                  #   A_DIM
                tf.nn.tanh,             #   tanh é o tipo de ativação da saída da camada, retorna um valor entre 1 e -1
                trainable=trainable,    #   trainable determina se a rede é treinável ou não
                name = 'mu_'+name       #   name é o nome da camada
            )                           #   O resultado é multiplicado por 2 para se adequar ao ambiente, que trabalha com um range 2 e -2.

            
            #   Calcula o desvio padrão, o range onde estará a possibilidade de ação    
            sigma = tf.layers.dense(    # Camada sigma do ATOR  
                l1,                     #   l1 é a entrada da camada
                A_DIM,                  #   A_DIM
                tf.nn.softplus,         #   softplus é o tipo de ativação da saída da camada 
                trainable=trainable,    #   trainable determina se a rede é treinável ou não
                name ='sigma_'+name     #   name é o nome da camada
            )    

            norm_dist = tf.distributions.Normal(    # Normaliza a saída mu da rede, considerando sigma
                loc=mu,                             # Loc é a média
                scale=sigma
            )            
                                                                                
        params = tf.get_collection(         # Coleta em params os pesos 
            tf.GraphKeys.GLOBAL_VARIABLES,  # das camadas l1,mu/2 e sigma
            scope=name                      # do escopo atual
        )   
        return norm_dist, params    # Retorna a ação e os pesos atuais das redes para serem armazenados na politica antiga.

    def choose_action(self, s):     # Recebe o estado s e retorna uma ação a
        s = s[np.newaxis, :]        #   Recebe o estado s e 
        a = self.sess.run(
            self.sample_op,         #   Executa sample_op 
            {self.tfs: s}           #   com o placeholder tfs que recebe o estado s e armazena a ação em a
        )[0]
        return np.clip(a, -2, 2)    #   Retorna um valor de ação a clipado entre -2 e 2

    def get_value(self, s):         # Recebe o estado s e retorna o valor da taxa de aprendizagem da CRITICA
        if s.ndim < 2: s = s[np.newaxis, :] 
        return self.sess.run(   # Retorna a taxa de aprendizagem da CRITICA
            self.v,             # v é a saída de valores da CRITICA
            {self.tfs: s}       # tfs é o placeholder que recebe o estado s
        )[0, 0] 