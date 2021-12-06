
class HostApp extends React.Component {

    state = {
        images: []
    };

    componentDidMount() {
        fetch("/images")
            .then(res => res.json()).then(
                (result) => {
                    this.setState({
                        images: result.images
                    });
                },
                (error) => {
                    console.log(error);
                });

    }

    render() {

        return (
            <div className="fill">
                <div className="d-flex h-100 text-center text-white bg-dark">
                    <svg id="root-canvas" className="fill"/>
                    <div id="root-ui" className="cover-container d-flex w-100 h-100 p-3 mx-auto flex-column">
                        <header className="mb-auto">
                            <div>
                                <h3 className="float-md-start mb-0">RFI Christmas Dream</h3>
                                <nav className="nav nav-masthead justify-content-center float-md-end">
                                    <a className="nav-link active" aria-current="page" href="#">Upload</a>
                                    <a className="nav-link" href="/about">About</a>
                                </nav>
                            </div>
                        </header>
                        <main className="px-3">
                            <div className="row">
                                {this.state.images.map((image) => <div className="col-xs-4"><img className="img-thumbnail" alt="" src={image}/></div>)}
                            </div>
                        </main>
                        <footer className="mt-auto text-white-50">
                            <p></p>
                        </footer>
                    </div>
                </div>
            </div>
        );
    }
}

ReactDOM.render(<HostApp />, document.getElementById('root'));
