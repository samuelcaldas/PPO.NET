using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;

namespace ppo.net
{
    class Program
    {
        static void Main(string[] args)
        {
            Test();
            Console.ReadKey();
        }

        private static void Test()
        {
            using (Py.GIL()) // Inicialize o mecanismo Python e adquira o bloqueio do interpretador
            {
                try
                {
                    //  Importações //
                    dynamic site = Py.Import("site");
                    dynamic sys = Py.Import("sys");
                    site.addsitedir(@"C:\Users\samue\source\repos\PPO.NET");
                    dynamic gym = Py.Import("gym");
                    dynamic np = Py.Import("numpy");    // Numpy para trabalhar com arrays
                    dynamic PPO = Py.Import("PPO");
                    dynamic plt = Py.Import("matplotlib.pyplot");

                    //  Configurações   //
                    int TRAINING_EPOCHS = 700;  // Quantidade total de episódios
                    int TRAINING_STEPS = 200;   // Quantas sequencias vão acontecer dentro de cada episódio
                    float GAMMA = (float)0.9;                   // Avanço (?)
                    int NUMBER_OF_SAMPLES = 64; // Tamanho do pacote à entrar para treinamento em cada etapa (?)

                    // Implementação do ambiente   //
                    dynamic env = gym.make("Pendulum-v0").unwrapped;        // Instancia o ambiente pendulo
                    dynamic ppo = PPO.PPO();                                // Instancia action classe PPO
                    List<double> all_epochs_rewards = new List<double>();   // Cria um array para action recompensa de todos os episódios

                    //  Loop de episódios //
                    for (int EPOCH = 0; EPOCH < TRAINING_EPOCHS; EPOCH++)   // TRAINING_EPOCHS: quantidade de episódios 
                    {
                        dynamic state = env.reset();                        // Redefine o ambiente e armazena o estado atual em state
                        // Cria quatro arrais para o episódio:
                        List<PyObject> buffered_states = new List<PyObject>();  // buffered_states: buffer do estado
                        List<PyObject> buffered_actions = new List<PyObject>(); // buffered_actions: buffer da ação
                        List<double> buffered_rewards = new List<double>();      // buffered_rewards: buffer da recompensa         
                        List<double> discounted_reward = new List<double>();    // Cria um array pra armazenar as recompensas calculadas
                        double epoch_reward = 0;    // Recompensa do episódio
                        //  Loop de episódio //
                        for (int STEP = 0; STEP < TRAINING_STEPS; STEP++)   // Duração de cada episodio
                        {
                            env.render();                           // Renderiza o ambiente
                            var action = ppo.choose_action(state);  // Envia um estado state e recebe uma ação action 
                            dynamic step = env.step(action);        // Envia uma ação action ao ambiente e recebe o estado step_state, e action recompensa reward
                            PyObject step_state = step.GetItem(0);  // Adiciona ao buffer de estado o estado atual state
                            double reward = (double)step.GetItem(1);
                            buffered_states.Add(state);             // Adiciona ao buffer de estado o estado atual state
                            buffered_actions.Add(action);           // Adiciona ao buffer de ação action ação atual action
                            buffered_rewards.Add((reward + 8) / 8);  // Adiciona ao buffer de recompensa action recompensa atual (?) normalizada (reward+8)/8
                            state = step_state;                     // Atualiza action variável de estado com o estado recebido pelo ambiente
                            epoch_reward += reward;                 // soma action recompensa da ação action recompensa do episodio

                            //  Atualiza PPO //
                            if ((STEP + 1) % NUMBER_OF_SAMPLES == 0 || STEP == TRAINING_STEPS - 1)
                            {
                                double value_state_ = ppo.get_v(step_state);    // Passa o estado atual step_state e recebe o valor atual da taxa de aprendizagem da CRITICA
                                                                                // V = learned state-value function
                                discounted_reward.Clear();                      // Limpa o array de recompensas calculadas
                                buffered_rewards.Reverse();                      // Coloca o array de buffer de recompensa ao contrario
                                foreach (double b_r in buffered_rewards)
                                {
                                    value_state_ = b_r + GAMMA * value_state_;  // Calcula action recompensa multiplicando action recompensa recebida reward pela GAMMA 
                                                                                // e pelo valor da taxa de aprendizado do estado value_state_
                                    discounted_reward.Add(value_state_);        // Adiciona ao array de recompensas calculadas 
                                }
                                discounted_reward.Reverse();                    // Coloca o array de recompensas calculadas ao contrario
                                // vstack transforma os arrays que estão em linha, em colunas
                                // Esses arrays de colunas são armazenados em _buffered_states _buffered_actions e _buffered_rewards
                                var _buffered_states = np.vstack(buffered_states);
                                var _buffered_actions = np.vstack(buffered_actions);
                                var _buffered_rewards = np.expand_dims(np.array(discounted_reward), axis: 1);
                                // Esvazia os buffers de estado, ação e recompensa
                                buffered_states.Clear();
                                buffered_actions.Clear();
                                buffered_rewards.Clear();
                                // Treine o cliente e o ator (status, ações, desconto de reward)
                                ppo.update( // Atualiza as redes com:
                                    _buffered_states,     //   Os estados acumulados
                                    _buffered_actions,     //   As ações acumuladas
                                    _buffered_rewards      //   As recompensas acumuladas
                                );
                            }
                        }
                        // Adiciona action recompensa do episodio atual ao array de recompensas
                        if (EPOCH == 0)
                        {
                            all_epochs_rewards.Add(epoch_reward);
                        }
                        else
                        {
                            all_epochs_rewards.Add(all_epochs_rewards.Last() * GAMMA + epoch_reward * 0.1);
                        }
                        // Escreve na tela
                        Console.WriteLine(
                            "Episódio: " + EPOCH +          // Numero do episodio
                            "   |   " +
                            "epoch_reward: " + epoch_reward // Recompensa do episodio
                        );
                    }
                    plt.plot(   // Plota o gráfico de todas as recompensas
                        np.arange(
                            all_epochs_rewards.Count
                        ),
                        all_epochs_rewards
                    );
                    plt.xlabel("Época");
                    plt.ylabel("Média móvel da recompensa de cada época");
                    plt.show();
                }
                //catch (PythonException error)
                catch (Exception error)
                {
                    // Comunique erros com exceções no script python -  isso funciona muito bem com pythonnet.
                    Console.WriteLine("Erro ", error.Message);
                }
            }
        }
    }
}
