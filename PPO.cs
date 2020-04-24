using Python.Runtime;
using System;
using System.Collections.Generic;

namespace ppo.net
{
    class PPO
    {
        // Classe PPO agrega:
        //   As redes neurais ATOR e CRITICA;
        //   Função para atualizar as redes neurais;
        //   Função para obter o valor de aprendizagem da CRITICA;
        //   Função para treinar as redes neurais;
        //   Função para escolher uma ação;

        // Importaçoes //
        public dynamic tf = Py.Import("tensorflow.compat.v1");// Workaround para retrocompatibilidade com tensorflow v1
        public dynamic np = Py.Import("numpy");               // Numpy para trabalhar com arrays

        // Configurações //

        double GAMMA = 0.9;             // Avanço (?)
        double A_LR = 0.0001;           // Taxa de aprendizado do ATOR
        double C_LR = 0.0002;           // Taxa de aprendizado da CRITICA
        int A_UPDATE_STEPS = 20;        // Quantidade de vezes que o treinamento do ATOR vai tomar a cadeia de
                                        // dados de batch
        int C_UPDATE_STEPS = 20;        // Quantidade de vezes que o treinamento da CRITICA vai tomar a cadeia de
                                        // dados de batch
        int S_DIM = 3;                  // S_DIM é a dimensao do estado, ou seja, quantas entradas ele
                                        // terá
        int A_DIM = 1;                  // A_DIM é a dimensão das ações, ou seja, quantas acões podem ser
                                        // executadas

        //int?[]
        private Dictionary<string, dynamic> METHOD = new Dictionary<string, dynamic>()
        {
                {"name", "clip"},   // Metodo de clip (Clipped surrogate objective) sujerido pelos papéis como mais eficiente
                {"epsilon", 0.2}    // epsilon=0.2 Valor de epsilon sujerido pelos papéis
        };
        private dynamic sess;
        private dynamic tfs;
        private dynamic tfa;
        private dynamic tfadv;
        private dynamic v;
        private dynamic tfdc_r;
        private dynamic advantage;
        private dynamic closs;
        private dynamic ctrain_op;
        private object pi;
        private object pi_parameters;
        private object oldpi;
        private object oldpi_parameters;

        PPO()
        { // Construtor da Classe
            tf.disable_v2_behavior();
            this.sess = tf.Session();    //inicializar uma seção do TensorFlow
                                         // Declaração das entradas das redes:
            this.tfs = tf.placeholder(// Estado do ambiente: a rede recebe o estado do ambiente através desse
                                      // placeholder
                tf.float32,             //   Tipo do placeholder

                new int?[] { null, S_DIM },          //   Dimensoes do placeholder
                "state"                 //   Nome do placeholder
            );

            this.tfa = tf.placeholder(      // Ação escolhida pela rede é informada através desse placeholder
                tf.float32,                 //   Tipo do placeholder

                new int?[] { null, A_DIM },              //   Dimensoes do placeholder
                "action"                    //   Nome do placeholder
            );

            this.tfadv = tf.placeholder(    // Calculo do ganho que a rede obteve no episódio, calculado fora da classe PPO.
                tf.float32,                 //   Tipo do placeholder

                new int?[] { null, 1 },                  //   Tamanho do placeholder
                "advantage"                 //   Nome do placeholder
            );                               // Esse placeholder é usado para treinar tanto o
                                             // ATOR quanto a CRITICA

            // CRITICA:

            // Criação da rede neural:
            var l1 = tf.layers.dense(       // Camada 1 entrada da Critica:
                this.tfs,               //   this.tfs é o placeholder do estado, funciona como entrada da
                                        //   rede
                    100,                    //   100 é o numero de neuronios
                    tf.nn.relu,             //   Relu é o tipo de ativação da saida da camada
                    name: "layer1-critic"    //   name é o nome da camada
                );

            this.v = tf.layers.dense(   // Camada de saida de valores da CRITICA:
                l1,                     //   l1 é a variavel referente a primeira camada da rede,
                1,                      //   1 é a quantidade de saidas da rede
                name: "V_layer"        //   name é o nome da camada
            );                           //   A saida dessa rede será o Q-Value, o status do
                                         //   progreço do aprendizado

            // Metodo de treinamento para o CRITICA, ou seja, o metodo de
            // aprendizagem:

            this.tfdc_r = tf.placeholder(   // A recompensa de cada episódio é inserida na rede através desse placeholder
                tf.float32,                 //   Tipo do placeholder


                new int?[]{ null, 1},                  //   Dimensoes do placeholder
                "discounted_r"              //   Nome do placeholder
            );
            this.advantage = this.tfdc_r - this.v   // Atraves da recompensa discounted_r/tfdc_r subtraida pelo
                                                    // valor de aprendizagem
                                                    // V_layer/v obtemos a vantagem
            this.closs = tf.reduce_mean(    // tf.reduce_mean calcula a média.
                tf.square(                  // tf.square calcula o quadrado
                    this.advantage          // da vantagem
                ));                          // !  Através disso obtemos em closs o Loss ou a
                                             // Perda da CRITICA

            this.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(this.closs);  // Ultilizamos o otimizador ADAM, com a taxa de aprendizado da CRITICA C_LR
                                                                                 // com a funçao minimize processamos os gradientes da CRITICA através da perda
                                                                                 // da CRITICA em closs
                                                                                 //   Poderiamos usar tambem o SGD como otimizador.


            // ATOR:
            //   Politica atual
            (pi, pi_parameters) = _build_anet("pi", trainable: true);                  // Criação da rede neural (pi) para a politica atual do ATOR
                                                                                            // através da função build_anet, definindo como treinavel
                                                                                            // pi é a saida da rede e pi_parameters são os pesos (estado atual)
                                                                                            // da rede
                                                                                            // Os pesos pi_parameters sao ultilizados para atualizar as politicas atual a antiga.


            this.sample_op = tf.squeeze(pi.sample(1), axis: 0);               // Tira uma amostra de açao da politica atual pi do ATOR


            //   Politica antiga
            (oldpi, oldpi_parameters) = this._build_anet("oldpi", trainable: false);    // Criação da rede neural oldpi para a politica antiga do ATOR através da
                                                                                        // função build_anet, definindo como não treinavel

            // Atualização dos pesos dos
            // pesos de oldpi tendo como
            // referencia os pesos de pi
            this.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_parameters, oldpi_parameters)]; // A cada atualização da rede, os pesos da politica atual passam para a
                                                                                                         // politica antiga
                                                                                                         // Update_oldpi_op
                                                                                                         // acumula
                                                                                                         // todos
                                                                                                         // os
                                                                                                         // valores
                                                                                                         // de
                                                                                                         // pi
                                                                                                         // ao
                                                                                                         // decorrer
                                                                                                         // do
                                                                                                         // episodio

                // Implementação da função de perda PPO
                // Funçao de perda:

                ratio = pi.prob(this.tfa) / oldpi.prob(this.tfa);    // O Ratio é a razão da probabilidade da ação tfa na politica nova
                                                                     // pela probabilidade da ação tfa na politica antiga.
                surr = ratio * this.tfadv                           // Surrogate é a Razão multiplicada pela vantagem


            this.aloss = -tf.reduce_mean(               // tf.educe_mean calcula a negativa da média do
                tf.minimum(                             //   menor valor entre
                    surr,                               //       o Surrogate e
                    this.tfadv *                        //       a multiplicação da vantagem
                        tf.clip_by_value(               //           pelo ratio clipado (limitado) por
                            ratio,                      //
                            1. - METHOD["epsilon"],     //                no maximo 1 - o metodo Clipped surrogate
                                                        //                objective
                            1. + METHOD["epsilon"]      //                no minimo 1 + o metodo Clipped surrogate
                                                        //                objective
                        )                               //
                )                                       // Obtendo assim em aloss a
                                                        // perda do Ator
            );

            // Metodo de treinamento para o ATOR, ou seja, o metodo de aprendizagem:

            this.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(this.aloss);  // Ultilizamos o otimizador ADAM, com a taxa de aprendizado do ATOR A_LR
                                                                                 // com minimize processamos os gradientes do ATOR através da perda do ATOR em aloss

            tf.summary.FileWriter("log/", this.sess.graph);      // Salvando o modelo na pasta log para analize futura no tensorboard

            this.sess.run(tf.global_variables_initializer());    // Inicializando todas as váriaveis definidas
        }

        private (object pi, object pi_parameters) _build_anet(string v, object trainable)
        {
            throw new NotImplementedException();
        }

        // Função de atualizaçao
        update(s, a, r)
        {              // Recebe o estado, a ação e a recompensa
            this.sess.run(this.update_oldpi_op); // Executa a matriz update_oldpi_op que comtem todos os pesos de pi/oldpi

            // Atualiza o ATOR
            adv = this.sess.run(this.advantage,     // Calcula a vantagem, ou seja, a recompensa do ATOR
            {
                this.tfs: s,    // Recebe o estado
                this.tfdc_r: r  // Recebe a recompensa
            });
        [this.sess.run(this.atrain_op,     // Treina o ator
            {
                this.tfs: s,    // Recebe o estado
                this.tfa: a,    // Recebe a ação
                this.tfadv: adv // Recebe o avanço
            }) for _ in range(A_UPDATE_STEPS)]   // A_UPDATE_STEPS é quantas vezes que a rede vai ser atualizada

        // Atualiza a CRITICA através da função de treinamento
        [this.sess.run(         // Executa o treinamento da critica
            this.ctrain_op,     //   ctrain_op é o treinamento da critica
            {
                this.tfs: s,    //   tfs é o placehoder que recebe estado s do ambiente
                this.tfdc_r: r  //   tfdc_r é o placeholder que recebe a recompensa r do ambiente
            }) for _ in range(C_UPDATE_STEPS)]   // C_UPDATE_STEPS é quantas vezes que a rede vai ser atualizada
                                                                                                        
                                                                                                        
                                                                                                        
                                                                                                        
}
     private void _build_anet(name, trainable)
{
    // Constroi as redes neurais do ATOR
    //    name é o nome da rede
    //    trainable determina se a rede é treinavel ou nao

    l1 = tf.layers.dense(       // Camada 1 entrada do ATOR:
        this.tfs,               //   this.tfs é o placeholder do estado, funciona como entrada
                                //   pra rede
        100,                    //   100 é o numero de neuronios
        tf.nn.relu,             //   Relu é o tipo de ativação da saida da rede
        trainable: trainable     //   trainable determina se a rede é treinavel ou nao
    );

    //   Calcula a ação que vai ser tomada
    mu = 2 * tf.layers.dense(   // Camada mu do ATOR
            l1,                     //   l1 é a entrada da camada
            A_DIM,                  //   A_DIM
            tf.nn.tanh,             //   tanh é o tipo de ativação da saida da camada, retorna um valor entre 1 e -1
            trainable: trainable,    //   trainable determina se a rede é treinavel ou nao
            name: "mu_" + name     //   name é o nome da camada
        );                           //   O resultado é multiplicado por 2 para se adequar
                                     //   ao ambiente, que trabalha com um range 2 e -2.


    // Calcula o desvio padrão, o range onde estará a possibilidade de
    // ação
    sigma = tf.layers.dense(    // Camada sigma do ATOR
            l1,                     //   l1 é a entrada da camada
            A_DIM,                  //   A_DIM
            tf.nn.softplus,         //   softplus é o tipo de ativação da saida da camada
            trainable: trainable,    //   trainable determina se a rede é treinavel ou nao
            name: "sigma_" + name   //   name é o nome da camada
        );

    norm_dist = tf.distributions.Normal(// Normaliza a saida mu da rede, considerando sigma
        loc: mu,                             // Loc é a média
        scale: sigma);

    parameters = tf.get_collection(             // Coleta em parameters os pesos
        tf.GraphKeys.GLOBAL_VARIABLES,      // das camadas l1,mu/2 e sigma
        scope: name                          // do scopo atual
    );
    return (norm_dist, parameters);    // Retorna a ação e os pesos atuais das redes para serem armazenados na
                                       // politica antiga.
}

void choose_action(s)
{     // Recebe o estado s e retorna uma ação a
    s = s[np.newaxis, :]        //   Recebe o estado s e
        a = this.sess.run(
            this.sample_op,         //   Executa sample_op
            { this.tfs: s}           //   com o placeholder tfs que recebe o estado s e armazena a açao em
                                    //   a
        )[0];    //
        return np.clip(a, -2, 2)    //   Retorna um valor de ação a clipado entre -2 e 2
    }
    void get_v(s)
{             // Recebe o estado s e retorna o valor da taxa de aprendizagem da
              // CRITICA
    if s.ndim < 2: s = s[np.newaxis, :] //
        return this.sess.run(   // Retorna a taxa de aprendizagem da CRITICA
            this.v,             // v é a saida de valores da CRITICA
            { this.tfs: s}       // tfs é o placeholder que recebe o estado s
        )[0, 0] //
    }
}
    }