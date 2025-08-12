#include <stdexcept>
#include <yaml-cpp/yaml.h>

int main()
{
    YAML::Node node = YAML::Load("[1, 2, 3]");
    if(!node.IsSequence()){
        throw std::runtime_error("Test failed");
    }
    return 0;
}
